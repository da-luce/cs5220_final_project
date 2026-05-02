// No-op stub for Intel VTune JIT profiling symbols.
//
// The PyTorch wheels distributed by NERSC are built with Intel VTune instrumentation
// enabled, so libtorch_cpu.so exports undefined references to iJIT_* symbols from the
// Intel Instrumentation and Tracing Technology (ITT) API.  VTune itself is not installed
// on Perlmutter compute nodes, so the dynamic linker fails at load time unless something
// provides these symbols.
//
// This file provides hollow implementations that satisfy the linker without pulling in
// the full VTune runtime.  Build it via -DBUILD_ITT_STUB=ON (done automatically by
// `make configure_perlmutter`).

extern "C" {
int iJIT_NotifyEvent(int event_type, void *event_data) { return 0; }
int iJIT_IsProfilingActive(void) { return 0; }
unsigned int iJIT_GetNewMethodID(void) { return 0; }
}
