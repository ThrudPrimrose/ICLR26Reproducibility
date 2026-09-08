subroutine ext_break_capture_fp64(a, out_index, out_value, LEN_1D, workspace, workspace_size) bind(C)
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D, workspace_size
  real(c_double), intent(in) :: a(LEN_1D)
  integer(c_int64_t), intent(inout) :: out_index(1)
  real(c_double), intent(inout) :: out_value(1)
  integer(c_int8_t), intent(inout) :: workspace(workspace_size)

  integer(c_int64_t) :: i, t, nt, lo, hi, first_idx, local_first
  integer(c_int64_t), allocatable :: locals(:)
  real(c_double), parameter :: K = 1.0d0

  nt = omp_get_max_threads()
  allocate(locals(nt))
  locals = LEN_1D + 1

  !$omp parallel private(t, lo, hi, local_first, i)
  t = omp_get_thread_num() + 1
  lo = ((LEN_1D * (t - 1)) / nt) + 1
  hi = (LEN_1D * t) / nt
  local_first = LEN_1D + 1
  do i = lo, hi
    if (a(i) > K) then
      local_first = i
      exit
    end if
  end do
  locals(t) = local_first
  !$omp end parallel

  first_idx = minval(locals)

  if (first_idx <= LEN_1D) then
    out_index(1) = first_idx
    out_value(1) = a(first_idx)
  else
    out_index(1) = 0
    out_value(1) = -1.0d0
  end if
end subroutine ext_break_capture_fp64
