/**
 * @file
 *
 * @brief Facilities for exception-based handling of Runtime
 * and Driver API errors, including a basic exception class
 * wrapping `::std::runtime_error`.
 *
 * @note Does not - for now - support wrapping errors generated
 * by other CUDA-related libraries like NVRTC.
 *
 * @note Unlike the Runtime API, the driver API has no memory
 * of "non-sticky" errors, which do not corrupt the current
 * context.
 *
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_ERROR_HPP_
#define CUDA_API_WRAPPERS_ERROR_HPP_

#include "types.hpp"
#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>

#include <type_traits>
#include <string>
#include <stdexcept>

namespace cuda {

namespace status {

/**
 * Aliases for CUDA status codes
 *
 * @note unfortunately, this enum can't inherit from @ref cuda::status_t
 */
enum named_t : ::std::underlying_type<status_t>::type {
	success                          = hipSuccess,
	memory_allocation_failure        = hipErrorOutOfMemory, // corresponds to hipErrorOutOfMemory
	not_yet_initialized              = hipErrorNotInitialized, // corresponds to hipErrorNotInitialized
	already_deinitialized            = hipErrorDeinitialized, // corresponds to hipErrorDeinitialized
	profiler_disabled                = hipErrorProfilerDisabled,
#if CUDA_VERSION >= 10100
	profiler_not_initialized         = hipErrorProfilerNotInitialized,
#endif
	profiler_already_started         = hipErrorProfilerAlreadyStarted,
	profiler_already_stopped         = hipErrorProfilerAlreadyStopped,
#if CUDA_VERSION >= 11100
	stub_library                     = CUDA_ERROR_STUB_LIBRARY,
	device_not_licensed              = CUDA_ERROR_DEVICE_NOT_LICENSED,
#endif
	prior_launch_failure             = hipErrorPriorLaunchFailure,
	launch_timeout                   = hipErrorLaunchTimeOut,
	launch_out_of_resources          = hipErrorLaunchOutOfResources,
	// kernel_launch_incompatible_texturing_mode = CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING,
	invalid_kernel_function          = hipErrorInvalidDeviceFunction,
	invalid_configuration            = hipErrorInvalidConfiguration,
	invalid_device                   = hipErrorInvalidDevice,
	invalid_value                    = hipErrorInvalidValue,
	invalid_pitch_value              = hipErrorInvalidPitchValue,
	invalid_symbol                   = hipErrorInvalidSymbol,
	map_buffer_object_failed         = hipErrorMapFailed, // corresponds to hipErrorMapFailed,
	unmap_buffer_object_failed       = hipErrorUnmapFailed, // corresponds to hipErrorUnmapFailed,
	array_still_mapped               = hipErrorArrayIsMapped,
	resource_already_mapped          = hipErrorAlreadyMapped,
	resource_already_acquired        = hipErrorAlreadyAcquired,
	resource_not_mapped              = hipErrorNotMapped,
	not_mapped_as_pointer            = hipErrorNotMappedAsPointer,
	not_mapped_as_array              = hipErrorNotMappedAsArray,
	// invalid_host_pointer             = cudaErrorInvalidHostPointer,
	invalid_device_pointer           = hipErrorInvalidDevicePointer,
	// invalid_texture                  = cudaErrorInvalidTexture,
	// invalid_texture_binding          = cudaErrorInvalidTextureBinding,
	// invalid_channel_descriptor       = cudaErrorInvalidChannelDescriptor,
	invalid_memcpy_direction         = hipErrorInvalidMemcpyDirection,
	// address_of_constant              = cudaErrorAddressOfConstant,
	// texture_fetch_failed             = cudaErrorTextureFetchFailed,
	// texture_not_bound                = cudaErrorTextureNotBound,
	// synchronization_error            = cudaErrorSynchronizationError,
	// invalid_filter_setting           = cudaErrorInvalidFilterSetting,
	// invalid_norm_setting             = cudaErrorInvalidNormSetting,
	// mixed_device_execution           = cudaErrorMixedDeviceExecution,
	unknown                          = hipErrorUnknown,
	// not_yet_implemented              = cudaErrorNotYetImplemented,
	// memory_value_too_large           = cudaErrorMemoryValueTooLarge,
	invalid_resource_handle          = hipErrorInvalidHandle,
#if CUDA_VERSION >= 10000
	resource_not_in_valid_state     = hipErrorIllegalState,
#endif
	async_operations_not_yet_completed = hipErrorNotReady,
	insufficient_driver              = hipErrorInsufficientDriver,
	set_on_active_process            = hipErrorSetOnActiveProcess,
	// invalid_surface                  = cudaErrorInvalidSurface,
	symbol_not_found                 = hipErrorNotFound, // corresponds to hipErrorNotFound
	no_device                        = hipErrorNoDevice,
	ecc_uncorrectable                = hipErrorECCNotCorrectable,
	shared_object_symbol_not_found   = hipErrorSharedObjectSymbolNotFound,
	invalid_source                   = hipErrorInvalidSource,
	file_not_found                   = hipErrorFileNotFound,
	shared_object_init_failed        = hipErrorSharedObjectInitFailed,
	// jit_compiler_not_found           = CUDA_ERROR_JIT_COMPILER_NOT_FOUND,
#if CUDA_VERSION >= 11100
	unsupported_ptx_version          = CUDA_ERROR_UNSUPPORTED_PTX_VERSION,
#endif
#if CUDA_VERSION >= 11200
	jit_compilation_disabled         = CUDA_ERROR_JIT_COMPILATION_DISABLED,
#endif
#if CUDA_VERSION >= 11400
	unsupported_exec_affinity        = CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY,
#endif
	unsupported_limit                = hipErrorUnsupportedLimit,
	// duplicate_variable_name          = cudaErrorDuplicateVariableName,
	// duplicate_texture_name           = cudaErrorDuplicateTextureName,
	// duplicate_surface_name           = cudaErrorDuplicateSurfaceName,
	// devices_unavailable              = cudaErrorDevicesUnavailable,
	invalid_kernel_image             = hipErrorInvalidImage, // corresponds to hipErrorInvalidImage,
	no_kernel_image_for_device       = hipErrorNoBinaryForGpu, // corresponds to hipErrorNoBinaryForGpu,
	// incompatible_driver_context      = cudaErrorIncompatibleDriverContext,
	missing_configuration            = hipErrorMissingConfiguration,
	invalid_context                  = hipErrorInvalidContext,
	context_already_current          = hipErrorContextAlreadyCurrent,
	context_already_in_use           = hipErrorContextAlreadyInUse,
	peer_access_already_enabled      = hipErrorPeerAccessAlreadyEnabled,
	peer_access_not_enabled          = hipErrorPeerAccessNotEnabled,
	device_already_in_use            = hipErrorContextAlreadyInUse,
	primary_context_already_active   = hipErrorSetOnActiveProcess,
	context_is_destroyed             = hipErrorContextIsDestroyed,
	primary_context_is_uninitialized = hipErrorContextIsDestroyed, // an alias!
#if CUDA_VERSION >= 10200
	device_uninitialized             = hipErrorInvalidContext,
#endif
	assert                           = hipErrorAssert,
	// too_many_peers                   = CUDA_ERROR_TOO_MANY_PEERS,
	host_memory_already_registered   = hipErrorHostMemoryAlreadyRegistered,
	host_memory_not_registered       = hipErrorHostMemoryNotRegistered,
	operating_system                 = hipErrorOperatingSystem,
	peer_access_unsupported          = hipErrorPeerAccessUnsupported,
	// launch_max_depth_exceeded        = cudaErrorLaunchMaxDepthExceeded,
	// launch_file_scoped_tex           = cudaErrorLaunchFileScopedTex,
	// launch_file_scoped_surf          = cudaErrorLaunchFileScopedSurf,
	// sync_depth_exceeded              = cudaErrorSyncDepthExceeded,
	// launch_pending_count_exceeded    = cudaErrorLaunchPendingCountExceeded,
	invalid_device_function          = hipErrorInvalidDeviceFunction,
	// not_permitted                    = CUDA_ERROR_NOT_PERMITTED,
	not_supported                    = hipErrorNotSupported,
	// hardware_stack_error             = CUDA_ERROR_HARDWARE_STACK_ERROR,
	// illegal_instruction              = CUDA_ERROR_ILLEGAL_INSTRUCTION,
	// misaligned_address               = CUDA_ERROR_MISALIGNED_ADDRESS,
	exception_during_kernel_execution = hipErrorLaunchFailure,
	cooperative_launch_too_large     = hipErrorCooperativeLaunchTooLarge,
	// invalid_address_space            = CUDA_ERROR_INVALID_ADDRESS_SPACE,
	// invalid_pc                       = CUDA_ERROR_INVALID_PC,
	illegal_address                  = hipErrorIllegalAddress,
	invalid_ptx                      = hipErrorInvalidKernelFile,
	invalid_graphics_context         = hipErrorInvalidGraphicsContext,
	// nvlink_uncorrectable             = CUDA_ERROR_NVLINK_UNCORRECTABLE,
	// startup_failure                  = cudaErrorStartupFailure,
	// api_failure_base                 = cudaErrorApiFailureBase,
#if CUDA_VERSION >= 10000
	system_not_ready                 = CUDA_ERROR_SYSTEM_NOT_READY,
#endif
#if CUDA_VERSION >= 10100
	system_driver_mismatch           = CUDA_ERROR_SYSTEM_DRIVER_MISMATCH,
	not_supported_on_device          = CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE,
#endif
#if CUDA_VERSION >= 10000
	stream_capture_unsupported       = hipErrorStreamCaptureUnsupported,
	stream_capture_invalidated       = hipErrorStreamCaptureInvalidated,
	stream_capture_merge             = hipErrorStreamCaptureMerge,
	stream_capture_unmatched         = hipErrorStreamCaptureUnmatched,
	stream_capture_unjoined          = hipErrorStreamCaptureUnjoined,
	stream_capture_isolation         = hipErrorStreamCaptureIsolation,
	stream_capture_disallowed_implicit_dependency = hipErrorStreamCaptureImplicit,
	not_permitted_on_captured_event  = hipErrorCapturedEvent,
#endif
#if CUDA_VERSION >= 10100
	stream_capture_wrong_thread      = hipErrorStreamCaptureWrongThread,
#endif
#if CUDA_VERSION >= 10200
	timeout_lapsed                   = CUDA_ERROR_TIMEOUT,
	graph_update_would_violate_constraints = hipErrorGraphExecUpdateFailure,
#endif
#if CUDA_VERSION >= 11400
	mps_connection_failed            = CUDA_ERROR_MPS_CONNECTION_FAILED,
	mps_rpc_failure                  = CUDA_ERROR_MPS_RPC_FAILURE,
	mps_server_not_ready             = CUDA_ERROR_MPS_SERVER_NOT_READY,
	mps_max_clients_reached          = CUDA_ERROR_MPS_MAX_CLIENTS_REACHED,
	mps_max_connections_reached      = CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED,
	async_error_in_external_device   = CUDA_ERROR_EXTERNAL_DEVICE,
#endif
};

///@cond
constexpr inline bool operator==(const status_t& lhs, const named_t& rhs) noexcept { return lhs == static_cast<status_t>(rhs); }
constexpr inline bool operator!=(const status_t& lhs, const named_t& rhs) noexcept { return lhs != static_cast<status_t>(rhs); }
constexpr inline bool operator==(const named_t& lhs, const status_t& rhs) noexcept { return static_cast<status_t>(lhs) == rhs; }
constexpr inline bool operator!=(const named_t& lhs, const status_t& rhs) noexcept { return static_cast<status_t>(lhs) != rhs; }
///@endcond

} // namespace status

/**
 * @brief Determine whether the API call returning the specified status had succeeded
 */
///@{
constexpr bool is_success(status_t status)  { return status == static_cast<status_t>(status::success); }
///@}

/**
 * @brief Determine whether the API call returning the specified status had failed
 */
///@{
constexpr bool is_failure(status_t status)  { return not is_success(status); }
///@}

/**
 * Obtain a brief textual explanation for a specified kind of CUDA Runtime API status
 * or error code.
 */
///@{
inline ::std::string describe(status_t status) { return hipGetErrorString(status); }
///@}


namespace detail_ {

template <typename I, bool UpperCase = false>
::std::string as_hex(I x)
{
	static_assert(::std::is_unsigned<I>::value, "only signed representations are supported");
	unsigned num_hex_digits = 2*sizeof(I);
	if (x == 0) return "0x0";

	enum { bits_per_hex_digit = 4 }; // = log_2 of 16
	static const char* digit_characters =
		UpperCase ? "0123456789ABCDEF" : "0123456789abcdef" ;

	::std::string result(num_hex_digits,'0');
	for (unsigned digit_index = 0; digit_index < num_hex_digits ; digit_index++)
	{
		size_t bit_offset = (num_hex_digits - 1 - digit_index) * bits_per_hex_digit;
		auto hexadecimal_digit = (x >> bit_offset) & 0xF;
		result[digit_index] = digit_characters[hexadecimal_digit];
	}
	return "0x0" + result.substr(result.find_first_not_of('0'), ::std::string::npos);
}

// TODO: Perhaps find a way to avoid the extra function, so that as_hex() can
// be called for pointer types as well? Would be easier with boost's uint<T>...
template <typename I, bool UpperCase = false>
inline ::std::string ptr_as_hex(const I* ptr)
{
	return as_hex(reinterpret_cast<uintptr_t>(ptr));
}

} // namespace detail_

/**
 * A (base?) class for exceptions raised by CUDA code; these errors are thrown by
 * essentially all CUDA Runtime API wrappers upon failure.
 *
 * A CUDA runtime error can be constructed with either just a CUDA error code
 * (=status code), or a code plus an additional message.
 *
 * @todo Consider renaming this to avoid confusion with the CUDA Runtime.
 */
class runtime_error : public ::std::runtime_error {
public:
	///@cond
	// TODO: Constructor chaining; and perhaps allow for more construction mechanisms?
	runtime_error(status_t error_code) :
		::std::runtime_error(describe(error_code)), code_(error_code)
	{ }
	// I wonder if I should do this the other way around
	runtime_error(status_t error_code, const ::std::string& what_arg) :
		::std::runtime_error(what_arg + ": " + describe(error_code)),
		code_(error_code)
	{ }
	// I wonder if I should do this the other way around
	runtime_error(status_t error_code, ::std::string&& what_arg) :
		runtime_error(error_code, what_arg)
	{ }
	///@endcond
	explicit runtime_error(status::named_t error_code) :
		runtime_error(static_cast<status_t>(error_code)) { }
	runtime_error(status::named_t error_code, const ::std::string& what_arg) :
		runtime_error(static_cast<status_t>(error_code), what_arg) { }
	runtime_error(status::named_t error_code, ::std::string&& what_arg) :
		runtime_error(static_cast<status_t>(error_code), what_arg) { }

protected:
	runtime_error(status_t error_code, ::std::runtime_error&& err) :
		::std::runtime_error(::std::move(err)), code_(error_code)
	{ }

public:
	static runtime_error with_message_override(status_t error_code, ::std::string complete_what_arg)
	{
		return runtime_error(error_code, ::std::runtime_error(::std::move(complete_what_arg)));
	}

	/**
	 * Obtain the CUDA status code which resulted in this error being thrown.
	 */
	status_t code() const { return code_; }

private:
	status_t code_;
};

#define throw_if_error_lazy(status__, ... ) \
do { \
	const ::cuda::status_t tie_status__ = static_cast<::cuda::status_t>(status__); \
	if (::cuda::is_failure(tie_status__)) { \
		throw ::cuda::runtime_error(tie_status__, (__VA_ARGS__)); \
	} \
} while(false)

// TODO: The following could use ::std::optional arguments - which would
// prevent the need for dual versions of the functions - but we're
// not writing C++17 here

/**
 * Do nothing... unless the status indicates an error, in which case
 * a @ref cuda::runtime_error exception is thrown
 *
 * @param status should be @ref cuda::status::success - otherwise an exception is thrown
 * @param message An extra description message to add to the exception
 */
inline void throw_if_error(status_t status, const ::std::string& message) noexcept(false)
{
	if (is_failure(status)) { throw runtime_error(status, message); }
}

inline void throw_if_error(status_t status, ::std::string&& message) noexcept(false)
{
	if (is_failure(status)) { throw runtime_error(status, message); }
}

/**
 * Does nothing - unless the status indicates an error, in which case
 * a @ref cuda::runtime_error exception is thrown
 *
 * @param status should be @ref cuda::status::success - otherwise an exception is thrown
 */
inline void throw_if_error(status_t status) noexcept(false)
{
	if (is_failure(status)) { throw runtime_error(status); }
}

enum : bool {
	dont_clear_errors = false,
	do_clear_errors    = true
};

namespace detail_ {

namespace outstanding_runtime_error {

/**
 * Clears the current CUDA context's status and return any outstanding error.
 *
 * @todo Reconsider what this does w.r.t. driver calls
 */
inline status_t clear() noexcept
{
	return static_cast<status_t>(hipGetLastError());
}

/**
 * Get the code of the last error in a CUDA-related action.
 *
 * @todo Reconsider what this does w.r.t. driver calls
 */
inline status_t get() noexcept
{
	return static_cast<status_t>(hipPeekAtLastError());
}

} // namespace outstanding_runtime_error
} // namespace detail_

/**
 * Unlike the Runtime API, where every error is outstanding
 * until cleared, the Driver API, which we use mostly, only
 * remembers "sticky" errors - severe errors which corrupt
 * contexts. Such errors cannot be recovered from / cleared,
 * and require either context destruction or process termination.
 */
namespace outstanding_error {

/**
 * @return the code of a sticky (= context-corrupting) error,
 * if the CUDA driver has recently encountered any.
 */
inline status_t get(bool try_clearing = false) noexcept(true)
{
	static constexpr const unsigned dummy_flags{0};
	auto status = hipInit(dummy_flags);
	if (not is_success(status)) { return status; }
	return static_cast<status_t>(try_clearing ? hipGetLastError() : hipPeekAtLastError());
}

/**
 * @brief Does nothing (unless throwing an exception)
 *
 * @note similar to @ref cuda::throw_if_error, but uses the CUDA driver's
 * own state regarding whether or not a sticky error has occurred
 */
inline void ensure_none(const ::std::string &message) noexcept(false)
{
	auto status = get();
	throw_if_error(status, message);
}

/**
 * @brief A variant of @ref ensure_none() which takes
 * a C-style string.
 *
 * @note exists so as to avoid incorrect overload resolution of
 * `ensure_none(my_c_string)` calls.
 */
inline void ensure_none(const char *message) noexcept(false)
{
	return ensure_none(::std::string{message});
}

/**
 * @brief Does nothing (unless throwing an exception)
 *
 * @note similar to @ref throw_if_error, but uses the CUDA Runtime API's internal
 * state
 *
 * @throws cuda::runtime_error if the CUDA runtime API has
 * encountered previously encountered an (uncleared) error
 *
 * @param clear_any_error When true, clears the CUDA Runtime API's state from
 * recalling errors arising from before this oment
 */
inline void ensure_none() noexcept(false)
{
	auto status = get();
	throw_if_error(status);
}

} // namespace outstanding_error

// The following few functions are used in the error messages
// generated for exceptions thrown by various API wrappers.

namespace device {
namespace detail_ {
inline ::std::string identify(device::id_t device_id)
{
	return ::std::string("device ") + ::std::to_string(device_id);
}
} // namespace detail_
} // namespace device

namespace context {
namespace detail_ {

inline ::std::string identify(handle_t handle)
{
	return "context " + cuda::detail_::ptr_as_hex(handle);
}

inline ::std::string identify(handle_t handle, device::id_t device_id)
{
	return identify(handle) + " on " + device::detail_::identify(device_id);
}

} // namespace detail_

namespace current {
namespace detail_ {
inline ::std::string identify(context::handle_t handle)
{
	return "current context: " + context::detail_::identify(handle);
}
inline ::std::string identify(context::handle_t handle, device::id_t device_id)
{
	return "current context: " + context::detail_::identify(handle, device_id);
}
} // namespace detail_
} // namespace current

} // namespace context

namespace device {
namespace primary_context {
namespace detail_ {

inline ::std::string identify(handle_t handle, device::id_t device_id)
{
	return "context " + context::detail_::identify(handle, device_id);
}
inline ::std::string identify(handle_t handle)
{
	return "context " + context::detail_::identify(handle);
}
} // namespace detail_
} // namespace primary_context
} // namespace device

namespace stream {
namespace detail_ {
inline ::std::string identify(handle_t handle)
{
	return "stream " + cuda::detail_::ptr_as_hex(handle);
}
inline ::std::string identify(handle_t handle, device::id_t device_id)
{
	return identify(handle) + " on " + device::detail_::identify(device_id);
}
inline ::std::string identify(handle_t handle, context::handle_t context_handle)
{
	return identify(handle) + " in " + context::detail_::identify(context_handle);
}
inline ::std::string identify(handle_t handle, context::handle_t context_handle, device::id_t device_id)
{
	return identify(handle) + " in " + context::detail_::identify(context_handle, device_id);
}
} // namespace detail_
} // namespace stream

namespace event {
namespace detail_ {
inline ::std::string identify(handle_t handle)
{
	return "event " + cuda::detail_::ptr_as_hex(handle);
}
inline ::std::string identify(handle_t handle, device::id_t device_id)
{
	return identify(handle) + " on " + device::detail_::identify(device_id);
}
inline ::std::string identify(handle_t handle, context::handle_t context_handle)
{
	return identify(handle) + " on " + context::detail_::identify(context_handle);
}
inline ::std::string identify(handle_t handle, context::handle_t context_handle, device::id_t device_id)
{
	return identify(handle) + " on " + context::detail_::identify(context_handle, device_id);
}
} // namespace detail_
} // namespace event

namespace kernel {
namespace detail_ {

inline ::std::string identify(const void* ptr)
{
	return "kernel " + cuda::detail_::ptr_as_hex(ptr);
}
inline ::std::string identify(const void* ptr, device::id_t device_id)
{
	return identify(ptr) + " on " + device::detail_::identify(device_id);
}
inline ::std::string identify(const void* ptr, context::handle_t context_handle)
{
	return identify(ptr) + " in " + context::detail_::identify(context_handle);
}
inline ::std::string identify(const void* ptr, context::handle_t context_handle, device::id_t device_id)
{
	return identify(ptr) + " in " + context::detail_::identify(context_handle, device_id);
}
inline ::std::string identify(handle_t handle)
{
	return "kernel at " + cuda::detail_::ptr_as_hex(handle);
}
inline ::std::string identify(handle_t handle, context::handle_t context_handle)
{
	return identify(handle) + " in " + context::detail_::identify(context_handle);
}
inline ::std::string identify(handle_t handle,  device::id_t device_id)
{
	return identify(handle) + " on " + device::detail_::identify(device_id);
}
inline ::std::string identify(handle_t handle, context::handle_t context_handle, device::id_t device_id)
{
	return identify(handle) + " in " + context::detail_::identify(context_handle, device_id);
}

} // namespace detail
} // namespace kernel

namespace memory {
namespace detail_ {

inline ::std::string identify(region_t region)
{
	return ::std::string("memory region at ") + cuda::detail_::ptr_as_hex(region.data())
		+ " of size " + ::std::to_string(region.size());
}

} // namespace detail_

} // namespace memory

} // namespace cuda

#endif // CUDA_API_WRAPPERS_ERROR_HPP_
