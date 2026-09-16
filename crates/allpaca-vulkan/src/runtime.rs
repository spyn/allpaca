//! Ash instance, device, allocator, and compute dispatch (linked SPIR-V).

use super::{degraded, DeviceSlice};
use allpaca_types::{AllpacaError, DeviceInfo, DeviceKind};
use ash::vk;
use ash::{Device, Entry, Instance};
use gpu_allocator::vulkan::{
    Allocation, AllocationCreateDesc, AllocationScheme, Allocator, AllocatorCreateDesc,
};
use gpu_allocator::MemoryLocation;
use std::ffi::{CStr, CString};
use std::mem::{size_of, ManuallyDrop};
use std::sync::{Arc, Mutex};

const SPV_GEMM: &[u8] = include_bytes!(env!("ALLPACA_SPV_GEMM_F32"));
const SPV_RESIDUAL: &[u8] = include_bytes!(env!("ALLPACA_SPV_RESIDUAL_ADD"));
const SPV_RMSNORM: &[u8] = include_bytes!(env!("ALLPACA_SPV_RMSNORM"));
const SPV_SILU: &[u8] = include_bytes!(env!("ALLPACA_SPV_SILU_MUL"));
const SPV_EMBED: &[u8] = include_bytes!(env!("ALLPACA_SPV_EMBED_GATHER"));
const SPV_ROPE: &[u8] = include_bytes!(env!("ALLPACA_SPV_ROPE"));
const SPV_ATTENTION: &[u8] = include_bytes!(env!("ALLPACA_SPV_ATTENTION"));

#[repr(C)]
struct GemmPush {
    m: u32,
    n: u32,
    k: u32,
    _pad: u32,
}
#[repr(C)]
struct LenPush {
    n: u32,
    _pad: u32,
}
#[repr(C)]
struct RmsPush {
    n: u32,
    eps: f32,
}
#[repr(C)]
struct EmbedPush {
    token: u32,
    dim: u32,
}
#[repr(C)]
struct RopePush {
    n_heads: u32,
    head_dim: u32,
    pos: u32,
    theta: f32,
}
#[repr(C)]
struct AttnPush {
    n_q: u32,
    n_kv: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    causal: u32,
    _pad0: u32,
    _pad1: u32,
}

struct Pipe {
    shader: vk::ShaderModule,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    dset_layout: vk::DescriptorSetLayout,
}

pub(crate) struct DeviceBuf {
    shared: Arc<Shared>,
    buffer: vk::Buffer,
    allocation: Option<Allocation>,
    size: u64,
}

pub(crate) struct Shared {
    _entry: Entry,
    instance: Instance,
    device: Device,
    allocator: ManuallyDrop<Mutex<Allocator>>,
    queue: vk::Queue,
    cmd_pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    desc_pool: Mutex<vk::DescriptorPool>,
    gemm: Pipe,
    residual: Pipe,
    rmsnorm: Pipe,
    silu: Pipe,
    embed: Pipe,
    rope: Pipe,
    attention: Pipe,
    dummy: Mutex<Option<(vk::Buffer, Allocation)>>,
}

pub(crate) struct Inner {
    #[allow(dead_code)]
    pub device_index: i32,
    pub shared: Arc<Shared>,
    submit: Mutex<()>,
}

fn vk_err(op: &str, e: vk::Result) -> AllpacaError {
    degraded(format!("Vulkan {op}: {e:?}"))
}

fn alloc_err(op: &str, e: impl std::fmt::Display) -> AllpacaError {
    degraded(format!("Vulkan {op}: {e}"))
}

fn spirv_words(bytes: &[u8]) -> Result<Vec<u32>, AllpacaError> {
    if bytes.len() < 4 || bytes.len() % 4 != 0 {
        return Err(degraded("SPIR-V module is empty or unaligned"));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn load_entry() -> Result<Entry, AllpacaError> {
    // SAFETY: `Entry::load` dlopens the Vulkan loader (`vulkan-1.dll` / `libvulkan`).
    unsafe { Entry::load() }.map_err(|e| degraded(format!("Vulkan loader: {e}")))
}

fn create_instance(entry: &Entry) -> Result<Instance, AllpacaError> {
    let app = CString::new("allpaca").map_err(|_| degraded("app name"))?;
    let engine = CString::new("allpaca").map_err(|_| degraded("engine name"))?;
    let app_info = vk::ApplicationInfo::default()
        .application_name(&app)
        .application_version(0)
        .engine_name(&engine)
        .engine_version(0)
        .api_version(vk::API_VERSION_1_3);
    let info = vk::InstanceCreateInfo::default().application_info(&app_info);
    // SAFETY: `app`/`engine` CStrings outlive `create_instance`.
    unsafe { entry.create_instance(&info, None) }.map_err(|e| vk_err("create_instance", e))
}

struct Phys {
    handle: vk::PhysicalDevice,
    compute_family: u32,
    name: String,
    memory_bytes: u64,
}

fn list_compute_devices(instance: &Instance) -> Result<Vec<Phys>, AllpacaError> {
    // SAFETY: instance is valid.
    let devices = unsafe { instance.enumerate_physical_devices() }
        .map_err(|e| vk_err("enumerate_physical_devices", e))?;
    let mut out = Vec::new();
    for handle in devices {
        // SAFETY: `handle` came from enumerate.
        let props = unsafe { instance.get_physical_device_properties(handle) };
        let queues = unsafe { instance.get_physical_device_queue_family_properties(handle) };
        let Some(family) = queues
            .iter()
            .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
        else {
            continue;
        };
        if vk::api_version_major(props.api_version) < 1
            || (vk::api_version_major(props.api_version) == 1
                && vk::api_version_minor(props.api_version) < 3)
        {
            continue;
        }
        let mut vulkan12 = vk::PhysicalDeviceVulkan12Features::default();
        let mut vulkan11 = vk::PhysicalDeviceVulkan11Features::default();
        let mut features2 = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut vulkan11)
            .push_next(&mut vulkan12);
        // SAFETY: `handle` came from enumerate; `features2` pNext chain is live.
        unsafe {
            instance.get_physical_device_features2(handle, &mut features2);
        }
        if vulkan12.vulkan_memory_model != vk::TRUE
            || vulkan11.variable_pointers != vk::TRUE
            || vulkan11.variable_pointers_storage_buffer != vk::TRUE
        {
            continue;
        }
        let mem = unsafe { instance.get_physical_device_memory_properties(handle) };
        let memory_bytes = (0..mem.memory_heap_count as usize)
            .map(|i| mem.memory_heaps[i].size)
            .max()
            .unwrap_or(0);
        let raw = props.device_name;
        let name = raw
            .iter()
            .take_while(|c| **c != 0)
            .map(|c| *c as u8 as char)
            .collect();
        out.push(Phys {
            handle,
            compute_family: family as u32,
            name,
            memory_bytes,
        });
    }
    Ok(out)
}

pub(crate) fn enumerate_devices() -> Vec<DeviceInfo> {
    let Ok(entry) = load_entry() else {
        return Vec::new();
    };
    let Ok(instance) = create_instance(&entry) else {
        return Vec::new();
    };
    let list = list_compute_devices(&instance).unwrap_or_default();
    let infos = list
        .into_iter()
        .enumerate()
        .map(|(id, p)| DeviceInfo {
            id: id as i32,
            kind: DeviceKind::VulkanGpu,
            name: p.name,
            memory_bytes: p.memory_bytes,
            bandwidth_gbps: 0.0,
            peers: Vec::new(),
        })
        .collect();
    // SAFETY: we created `instance` and no device remains.
    unsafe {
        instance.destroy_instance(None);
    }
    infos
}

pub(crate) fn device_count() -> i32 {
    enumerate_devices().len() as i32
}

fn make_pipe(
    device: &Device,
    spv: &[u8],
    entry: &CStr,
    n_storage: u32,
    push_bytes: u32,
) -> Result<Pipe, AllpacaError> {
    let words = spirv_words(spv)?;
    let sm_info = vk::ShaderModuleCreateInfo::default().code(&words);
    // SAFETY: `words` is a valid SPIR-V word stream for this call.
    let shader = unsafe { device.create_shader_module(&sm_info, None) }
        .map_err(|e| vk_err("shader_module", e))?;
    let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..n_storage)
        .map(|i| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(i)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        })
        .collect();
    let dsl_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    // SAFETY: `bindings` live for this call.
    let dset_layout =
        unsafe { device.create_descriptor_set_layout(&dsl_info, None) }.map_err(|e| {
            // SAFETY: `shader` was created on this device and has not been used.
            unsafe {
                device.destroy_shader_module(shader, None);
            }
            vk_err("descriptor_set_layout", e)
        })?;
    let layouts = [dset_layout];
    let range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .offset(0)
        .size(push_bytes.max(4));
    let ranges = [range];
    let mut pl_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts);
    if push_bytes > 0 {
        pl_info = pl_info.push_constant_ranges(&ranges);
    }
    // SAFETY: `layouts` / optional push ranges live for this call.
    let layout = unsafe { device.create_pipeline_layout(&pl_info, None) }.map_err(|e| {
        // SAFETY: layout objects created so far belong to this device.
        unsafe {
            device.destroy_descriptor_set_layout(dset_layout, None);
            device.destroy_shader_module(shader, None);
        }
        vk_err("pipeline_layout", e)
    })?;
    let stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(shader)
        .name(entry);
    let ci = vk::ComputePipelineCreateInfo::default()
        .stage(stage)
        .layout(layout);
    // SAFETY: `stage` references `shader`/`entry`; `ci` lives for this call.
    let pipelines =
        unsafe { device.create_compute_pipelines(vk::PipelineCache::null(), &[ci], None) }
            .map_err(|(_, e)| {
                // SAFETY: pipeline layout objects created so far belong to this device.
                unsafe {
                    device.destroy_pipeline_layout(layout, None);
                    device.destroy_descriptor_set_layout(dset_layout, None);
                    device.destroy_shader_module(shader, None);
                }
                vk_err("compute_pipeline", e)
            })?;
    if pipelines.is_empty() {
        // SAFETY: no pipeline handle was returned; remaining objects are ours.
        unsafe {
            device.destroy_pipeline_layout(layout, None);
            device.destroy_descriptor_set_layout(dset_layout, None);
            device.destroy_shader_module(shader, None);
        }
        return Err(degraded(format!(
            "Vulkan compute_pipeline empty for {}",
            entry.to_string_lossy()
        )));
    }
    Ok(Pipe {
        shader,
        layout,
        pipeline: pipelines[0],
        dset_layout,
    })
}

fn destroy_pipe(device: &Device, pipe: &Pipe) {
    // SAFETY: `pipe` was created on `device` and is not referenced after this call.
    unsafe {
        device.destroy_pipeline(pipe.pipeline, None);
        device.destroy_pipeline_layout(pipe.layout, None);
        device.destroy_descriptor_set_layout(pipe.dset_layout, None);
        device.destroy_shader_module(pipe.shader, None);
    }
}

impl Inner {
    pub(crate) fn new(device_index: i32) -> Result<Self, AllpacaError> {
        let entry = load_entry()?;
        let instance = create_instance(&entry)?;
        let list = list_compute_devices(&instance)?;
        let phys = list.get(device_index as usize).ok_or_else(|| {
            // SAFETY: instance owned locally on failure.
            unsafe {
                instance.destroy_instance(None);
            }
            degraded("Vulkan workspace: no device")
        })?;
        let queue_family = phys.compute_family;
        let physical = phys.handle;
        let prio = [1.0f32];
        let qinfo = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&prio);
        let qinfos = [qinfo];
        let mut q12 = vk::PhysicalDeviceVulkan12Features::default();
        let mut q11 = vk::PhysicalDeviceVulkan11Features::default();
        let mut qfeat = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut q11)
            .push_next(&mut q12);
        // SAFETY: `physical` is a listed compute device; pNext chain is live.
        unsafe {
            instance.get_physical_device_features2(physical, &mut qfeat);
        }
        let core = qfeat.features;
        let mut vulkan12 = vk::PhysicalDeviceVulkan12Features::default()
            .vulkan_memory_model(true)
            .vulkan_memory_model_device_scope(true)
            .shader_int8(q12.shader_int8 == vk::TRUE)
            .storage_buffer8_bit_access(q12.storage_buffer8_bit_access == vk::TRUE)
            .uniform_and_storage_buffer8_bit_access(
                q12.uniform_and_storage_buffer8_bit_access == vk::TRUE,
            );
        let mut vulkan11 = vk::PhysicalDeviceVulkan11Features::default()
            .variable_pointers(true)
            .variable_pointers_storage_buffer(true)
            .storage_buffer16_bit_access(q11.storage_buffer16_bit_access == vk::TRUE)
            .uniform_and_storage_buffer16_bit_access(
                q11.uniform_and_storage_buffer16_bit_access == vk::TRUE,
            );
        let mut features2 = vk::PhysicalDeviceFeatures2::default()
            .features(
                vk::PhysicalDeviceFeatures::default()
                    .shader_int16(core.shader_int16 == vk::TRUE)
                    .shader_int64(core.shader_int64 == vk::TRUE),
            )
            .push_next(&mut vulkan11)
            .push_next(&mut vulkan12);
        let dinfo = vk::DeviceCreateInfo::default()
            .queue_create_infos(&qinfos)
            .push_next(&mut features2);
        // SAFETY: feature structs outlive `create_device`.
        let device = unsafe { instance.create_device(physical, &dinfo, None) }.map_err(|e| {
            unsafe {
                instance.destroy_instance(None);
            }
            vk_err("create_device", e)
        })?;
        let queue = unsafe { device.get_device_queue(queue_family, 0) };
        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device: physical,
            debug_settings: Default::default(),
            buffer_device_address: false,
            allocation_sizes: Default::default(),
        })
        .map_err(|e| {
            unsafe {
                device.destroy_device(None);
                instance.destroy_instance(None);
            }
            alloc_err("allocator", e)
        })?;

        let specs: [(&[u8], &CStr, u32, u32); 7] = [
            (SPV_GEMM, c"gemm_f32", 3, size_of::<GemmPush>() as u32),
            (
                SPV_RESIDUAL,
                c"residual_add",
                3,
                size_of::<LenPush>() as u32,
            ),
            (SPV_RMSNORM, c"rmsnorm", 3, size_of::<RmsPush>() as u32),
            (SPV_SILU, c"silu_mul", 3, size_of::<LenPush>() as u32),
            (SPV_EMBED, c"embed_gather", 2, size_of::<EmbedPush>() as u32),
            (SPV_ROPE, c"rope", 1, size_of::<RopePush>() as u32),
            (SPV_ATTENTION, c"attention", 4, size_of::<AttnPush>() as u32),
        ];
        let mut pipes: Vec<Pipe> = Vec::with_capacity(specs.len());
        for (spv, name, n_storage, push) in specs {
            match make_pipe(&device, spv, name, n_storage, push) {
                Ok(pipe) => pipes.push(pipe),
                Err(e) => {
                    for pipe in &pipes {
                        destroy_pipe(&device, pipe);
                    }
                    drop(allocator);
                    unsafe {
                        device.destroy_device(None);
                        instance.destroy_instance(None);
                    }
                    return Err(e);
                }
            }
        }
        let pipes: [Pipe; 7] = match pipes.try_into() {
            Ok(arr) => arr,
            Err(leftover) => {
                for pipe in leftover {
                    destroy_pipe(&device, &pipe);
                }
                drop(allocator);
                unsafe {
                    device.destroy_device(None);
                    instance.destroy_instance(None);
                }
                return Err(degraded("internal: expected 7 vulkan pipelines"));
            }
        };
        let [gemm, residual, rmsnorm, silu, embed, rope, attention] = pipes;

        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let cmd_pool = unsafe { device.create_command_pool(&pool_info, None) }
            .map_err(|e| vk_err("command_pool", e))?;
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(cmd_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd = unsafe { device.allocate_command_buffers(&alloc_info) }
            .map_err(|e| vk_err("command_buffer", e))?[0];
        let fence = unsafe {
            device.create_fence(
                &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                None,
            )
        }
        .map_err(|e| vk_err("fence", e))?;
        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(64)];
        let dp_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(16)
            .pool_sizes(&pool_sizes)
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET);
        let desc_pool = unsafe { device.create_descriptor_pool(&dp_info, None) }
            .map_err(|e| vk_err("descriptor_pool", e))?;

        let dummy_info = vk::BufferCreateInfo::default()
            .size(256)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let dummy_buf = unsafe { device.create_buffer(&dummy_info, None) }
            .map_err(|e| vk_err("dummy_buffer", e))?;
        let req = unsafe { device.get_buffer_memory_requirements(dummy_buf) };
        let mut allocator = allocator;
        let dummy_alloc = allocator
            .allocate(&AllocationCreateDesc {
                name: "dummy",
                requirements: req,
                location: MemoryLocation::GpuOnly,
                linear: true,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .map_err(|e| alloc_err("dummy alloc", e))?;
        unsafe {
            device
                .bind_buffer_memory(dummy_buf, dummy_alloc.memory(), dummy_alloc.offset())
                .map_err(|e| vk_err("bind dummy", e))?;
        }

        let shared = Arc::new(Shared {
            _entry: entry,
            instance,
            device,
            allocator: ManuallyDrop::new(Mutex::new(allocator)),
            queue,
            cmd_pool,
            cmd,
            fence,
            desc_pool: Mutex::new(desc_pool),
            gemm,
            residual,
            rmsnorm,
            silu,
            embed,
            rope,
            attention,
            dummy: Mutex::new(Some((dummy_buf, dummy_alloc))),
        });
        Ok(Self {
            device_index,
            shared,
            submit: Mutex::new(()),
        })
    }

    pub(crate) fn alloc(&self, size: u64, name: &str) -> Result<DeviceBuf, AllpacaError> {
        let size = size.max(256);
        let info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe { self.shared.device.create_buffer(&info, None) }
            .map_err(|e| vk_err("create_buffer", e))?;
        let req = unsafe { self.shared.device.get_buffer_memory_requirements(buffer) };
        let mut alloc = self
            .shared
            .allocator
            .lock()
            .map_err(|_| degraded("allocator mutex poisoned"))?;
        let allocation = alloc
            .allocate(&AllocationCreateDesc {
                name,
                requirements: req,
                location: MemoryLocation::CpuToGpu,
                linear: true,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .map_err(|e| {
                unsafe {
                    self.shared.device.destroy_buffer(buffer, None);
                }
                alloc_err("allocate", e)
            })?;
        unsafe {
            self.shared
                .device
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
                .map_err(|e| vk_err("bind_buffer", e))?;
        }
        Ok(DeviceBuf {
            shared: Arc::clone(&self.shared),
            buffer,
            allocation: Some(allocation),
            size,
        })
    }

    pub(crate) fn slice_of(buf: &DeviceBuf, offset: u64, bytes: u64) -> DeviceSlice {
        DeviceSlice {
            offset,
            bytes,
            buffer: buf.buffer,
        }
    }

    fn mapped_copy(buf: &DeviceBuf, offset: u64, src: &[u8]) -> Result<(), AllpacaError> {
        if src.is_empty() {
            return Ok(());
        }
        if offset as usize + src.len() > buf.size as usize {
            return Err(degraded("Vulkan mapped copy out of range"));
        }
        let Some(alloc) = buf.allocation.as_ref() else {
            return Err(degraded("Vulkan buffer has no allocation"));
        };
        let Some(ptr) = alloc.mapped_ptr() else {
            return Err(degraded("Vulkan buffer is not host-visible"));
        };
        // SAFETY: mapped CpuToGpu allocation, offset+len within `buf.size`.
        unsafe {
            std::ptr::copy_nonoverlapping(
                src.as_ptr(),
                ptr.as_ptr().cast::<u8>().add(offset as usize),
                src.len(),
            );
        }
        Ok(())
    }

    fn mapped_read(buf: &DeviceBuf, offset: u64, dst: &mut [u8]) -> Result<(), AllpacaError> {
        if dst.is_empty() {
            return Ok(());
        }
        if offset as usize + dst.len() > buf.size as usize {
            return Err(degraded("Vulkan mapped read out of range"));
        }
        let Some(alloc) = buf.allocation.as_ref() else {
            return Err(degraded("Vulkan buffer has no allocation"));
        };
        let Some(ptr) = alloc.mapped_ptr() else {
            return Err(degraded("Vulkan buffer is not host-visible"));
        };
        // SAFETY: mapped allocation after GPU idle; offset+len in range.
        unsafe {
            std::ptr::copy_nonoverlapping(
                ptr.as_ptr().cast::<u8>().add(offset as usize),
                dst.as_mut_ptr(),
                dst.len(),
            );
        }
        Ok(())
    }

    pub(crate) fn upload_f32(&self, host: &[f32]) -> Result<DeviceBuf, AllpacaError> {
        let bytes = (host.len() * 4) as u64;
        let buf = self.alloc(bytes.max(256), "upload")?;
        // SAFETY: `f32` is POD; the byte length is `len * 4`.
        let raw = unsafe { std::slice::from_raw_parts(host.as_ptr().cast::<u8>(), host.len() * 4) };
        Self::mapped_copy(&buf, 0, raw)?;
        Ok(buf)
    }

    pub(crate) fn download_f32(
        &self,
        buf: &DeviceBuf,
        dst: &mut [f32],
    ) -> Result<(), AllpacaError> {
        self.wait()?;
        // SAFETY: `dst` is a valid `f32` slice; we write `len * 4` bytes.
        let raw =
            unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr().cast::<u8>(), dst.len() * 4) };
        Self::mapped_read(buf, 0, raw)
    }

    pub(crate) fn write_into(
        &self,
        buf: &DeviceBuf,
        offset: u64,
        src: &[f32],
    ) -> Result<(), AllpacaError> {
        // SAFETY: `f32` is POD; the byte length is `len * 4`.
        let raw = unsafe { std::slice::from_raw_parts(src.as_ptr().cast::<u8>(), src.len() * 4) };
        Self::mapped_copy(buf, offset, raw)
    }

    fn wait(&self) -> Result<(), AllpacaError> {
        let _g = self
            .submit
            .lock()
            .map_err(|_| degraded("submit mutex poisoned"))?;
        // SAFETY: `fence` was created on `device` and is not in use by another waiter.
        unsafe {
            self.shared
                .device
                .wait_for_fences(&[self.shared.fence], true, 10_000_000_000)
                .map_err(|e| vk_err("wait_fences", e))?;
        }
        Ok(())
    }

    fn dispatch(
        &self,
        pipe: &Pipe,
        slices: &[DeviceSlice],
        push: &[u8],
        gx: u32,
        gy: u32,
        gz: u32,
    ) -> Result<(), AllpacaError> {
        let _g = self
            .submit
            .lock()
            .map_err(|_| degraded("submit mutex poisoned"))?;
        let device = &self.shared.device;
        unsafe {
            device
                .wait_for_fences(&[self.shared.fence], true, 10_000_000_000)
                .map_err(|e| vk_err("wait_fences", e))?;
            device
                .reset_fences(&[self.shared.fence])
                .map_err(|e| vk_err("reset_fences", e))?;
        }
        let pool = *self
            .shared
            .desc_pool
            .lock()
            .map_err(|_| degraded("desc pool mutex poisoned"))?;
        unsafe {
            device
                .reset_descriptor_pool(pool, vk::DescriptorPoolResetFlags::empty())
                .map_err(|e| vk_err("reset_descriptor_pool", e))?;
        }
        let alloc = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(std::slice::from_ref(&pipe.dset_layout));
        let set = unsafe { device.allocate_descriptor_sets(&alloc) }
            .map_err(|e| vk_err("allocate_descriptor_sets", e))?[0];
        let infos: Vec<vk::DescriptorBufferInfo> = slices
            .iter()
            .map(|s| {
                vk::DescriptorBufferInfo::default()
                    .buffer(s.buffer)
                    .offset(s.offset)
                    .range(s.bytes.max(4))
            })
            .collect();
        let writes: Vec<vk::WriteDescriptorSet> = infos
            .iter()
            .enumerate()
            .map(|(i, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(i as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(info))
            })
            .collect();
        unsafe {
            device.update_descriptor_sets(&writes, &[]);
        }

        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            device
                .begin_command_buffer(self.shared.cmd, &begin)
                .map_err(|e| vk_err("begin_command_buffer", e))?;
            device.cmd_bind_pipeline(
                self.shared.cmd,
                vk::PipelineBindPoint::COMPUTE,
                pipe.pipeline,
            );
            device.cmd_bind_descriptor_sets(
                self.shared.cmd,
                vk::PipelineBindPoint::COMPUTE,
                pipe.layout,
                0,
                &[set],
                &[],
            );
            if !push.is_empty() {
                device.cmd_push_constants(
                    self.shared.cmd,
                    pipe.layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    push,
                );
            }
            device.cmd_dispatch(self.shared.cmd, gx.max(1), gy.max(1), gz.max(1));
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::HOST_READ);
            device.cmd_pipeline_barrier(
                self.shared.cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
            device
                .end_command_buffer(self.shared.cmd)
                .map_err(|e| vk_err("end_command_buffer", e))?;
            let cmds = [self.shared.cmd];
            let submit = vk::SubmitInfo::default().command_buffers(&cmds);
            device
                .queue_submit(self.shared.queue, &[submit], self.shared.fence)
                .map_err(|e| vk_err("queue_submit", e))?;
            device
                .wait_for_fences(&[self.shared.fence], true, 10_000_000_000)
                .map_err(|e| vk_err("wait_fences", e))?;
        }
        Ok(())
    }

    pub(crate) fn gemm_nt(
        &self,
        a: DeviceSlice,
        b: DeviceSlice,
        c: DeviceSlice,
        m: i32,
        n: i32,
        k: i32,
    ) -> Result<(), AllpacaError> {
        if m <= 0 || n <= 0 || k <= 0 {
            return Err(degraded("invalid Vulkan gemm_nt"));
        }
        let push = GemmPush {
            m: m as u32,
            n: n as u32,
            k: k as u32,
            _pad: 0,
        };
        let gx = (n as u32 + 15) / 16;
        let gy = (m as u32 + 15) / 16;
        self.dispatch(&self.shared.gemm, &[a, b, c], bytes_of(&push), gx, gy, 1)
    }

    pub(crate) fn residual_add(
        &self,
        x: DeviceSlice,
        y: DeviceSlice,
        out: DeviceSlice,
        n: i32,
    ) -> Result<(), AllpacaError> {
        if n <= 0 {
            return Err(degraded("invalid Vulkan residual_add"));
        }
        let push = LenPush {
            n: n as u32,
            _pad: 0,
        };
        let gx = (n as u32 + 255) / 256;
        self.dispatch(
            &self.shared.residual,
            &[x, y, out],
            bytes_of(&push),
            gx,
            1,
            1,
        )
    }

    pub(crate) fn rmsnorm(
        &self,
        x: DeviceSlice,
        w: DeviceSlice,
        out: DeviceSlice,
        n: i32,
        eps: f32,
    ) -> Result<(), AllpacaError> {
        if n <= 0 {
            return Err(degraded("invalid Vulkan rmsnorm"));
        }
        let push = RmsPush { n: n as u32, eps };
        self.dispatch(&self.shared.rmsnorm, &[x, w, out], bytes_of(&push), 1, 1, 1)
    }

    pub(crate) fn silu_mul(
        &self,
        gate: DeviceSlice,
        up: DeviceSlice,
        out: DeviceSlice,
        n: i32,
    ) -> Result<(), AllpacaError> {
        if n <= 0 {
            return Err(degraded("invalid Vulkan silu_mul"));
        }
        let push = LenPush {
            n: n as u32,
            _pad: 0,
        };
        let gx = (n as u32 + 255) / 256;
        self.dispatch(
            &self.shared.silu,
            &[gate, up, out],
            bytes_of(&push),
            gx,
            1,
            1,
        )
    }

    pub(crate) fn embed_gather(
        &self,
        table: DeviceSlice,
        token: i32,
        out: DeviceSlice,
        dim: i32,
    ) -> Result<(), AllpacaError> {
        if dim <= 0 || token < 0 {
            return Err(degraded("invalid Vulkan embed_gather"));
        }
        let push = EmbedPush {
            token: token as u32,
            dim: dim as u32,
        };
        let gx = (dim as u32 + 255) / 256;
        self.dispatch(&self.shared.embed, &[table, out], bytes_of(&push), gx, 1, 1)
    }

    pub(crate) fn rope(
        &self,
        x: DeviceSlice,
        n_heads: i32,
        head_dim: i32,
        pos: i32,
        theta: f32,
    ) -> Result<(), AllpacaError> {
        if n_heads <= 0 || head_dim <= 0 || head_dim % 2 != 0 {
            return Err(degraded("invalid Vulkan rope"));
        }
        let push = RopePush {
            n_heads: n_heads as u32,
            head_dim: head_dim as u32,
            pos: pos as u32,
            theta,
        };
        let n_pairs = n_heads as u32 * (head_dim as u32 / 2);
        let gx = (n_pairs + 255) / 256;
        self.dispatch(&self.shared.rope, &[x], bytes_of(&push), gx, 1, 1)
    }

    pub(crate) fn attention(
        &self,
        q: DeviceSlice,
        k: DeviceSlice,
        v: DeviceSlice,
        out: DeviceSlice,
        n_q: i32,
        n_kv: i32,
        n_heads: i32,
        n_kv_heads: i32,
        head_dim: i32,
        causal: bool,
    ) -> Result<(), AllpacaError> {
        if n_q <= 0
            || n_kv <= 0
            || n_heads <= 0
            || n_kv_heads <= 0
            || head_dim <= 0
            || n_heads % n_kv_heads != 0
        {
            return Err(degraded("invalid Vulkan attention"));
        }
        let push = AttnPush {
            n_q: n_q as u32,
            n_kv: n_kv as u32,
            n_heads: n_heads as u32,
            n_kv_heads: n_kv_heads as u32,
            head_dim: head_dim as u32,
            causal: u32::from(causal),
            _pad0: 0,
            _pad1: 0,
        };
        self.dispatch(
            &self.shared.attention,
            &[q, k, v, out],
            bytes_of(&push),
            n_q as u32 * n_heads as u32,
            1,
            1,
        )
    }
}

fn bytes_of<T>(v: &T) -> &[u8] {
    // SAFETY: T is #[repr(C)] push-constant POD; read as bytes for vkCmdPushConstants.
    unsafe { std::slice::from_raw_parts((v as *const T).cast::<u8>(), size_of::<T>()) }
}

impl Drop for DeviceBuf {
    fn drop(&mut self) {
        if let Some(allocation) = self.allocation.take() {
            if let Ok(mut alloc) = self.shared.allocator.lock() {
                let _ = alloc.free(allocation);
            }
        }
        // SAFETY: buffer was created on this device and is no longer bound after `free`.
        unsafe {
            self.shared.device.destroy_buffer(self.buffer, None);
        }
    }
}

impl Drop for Shared {
    fn drop(&mut self) {
        // SAFETY: this is the last owner; no in-flight submits remain after wait idle.
        unsafe {
            let _ = self.device.device_wait_idle();
        }
        if let Ok(mut dummy) = self.dummy.lock() {
            if let Some((buf, allocation)) = dummy.take() {
                if let Ok(mut alloc) = self.allocator.lock() {
                    let _ = alloc.free(allocation);
                }
                unsafe {
                    self.device.destroy_buffer(buf, None);
                }
            }
        }
        destroy_pipe(&self.device, &self.gemm);
        destroy_pipe(&self.device, &self.residual);
        destroy_pipe(&self.device, &self.rmsnorm);
        destroy_pipe(&self.device, &self.silu);
        destroy_pipe(&self.device, &self.embed);
        destroy_pipe(&self.device, &self.rope);
        destroy_pipe(&self.device, &self.attention);
        // SAFETY: gpu-allocator must be dropped while the VkDevice is still alive.
        unsafe {
            ManuallyDrop::drop(&mut self.allocator);
        }
        // SAFETY: pipelines, pools, device, and instance were created on this Shared
        // and no DeviceBuf remains (those hold Arc clones).
        unsafe {
            if let Ok(pool) = self.desc_pool.lock() {
                self.device.destroy_descriptor_pool(*pool, None);
            }
            self.device.destroy_fence(self.fence, None);
            self.device.destroy_command_pool(self.cmd_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}
