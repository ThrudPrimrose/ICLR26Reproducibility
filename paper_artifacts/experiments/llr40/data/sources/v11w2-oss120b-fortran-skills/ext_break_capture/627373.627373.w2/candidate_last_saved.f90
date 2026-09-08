subroutine ext_break_capture_fp64(a, out_index, out_value, LEN_1D) bind(C)
  use iso_c_binding
  use omp_lib
  implicit none
  ! Arguments
  integer(c_int64_t), value, intent(in) :: LEN_1D
  real(c_double), intent(in) :: a(*)
  integer(c_int64_t), intent(out) :: out_index(*)
  real(c_double), intent(out) :: out_value(*)

  ! Local variables
  integer(c_int64_t) :: i, start, endi, tid, num_threads
  real(c_double), parameter :: k = 1.0_c_double
  integer(c_int64_t) :: local_idx
  real(c_double) :: local_val
  integer(c_int64_t), allocatable :: thread_idx(:)
  real(c_double), allocatable :: thread_val(:)

  ! Initialise output to sentinel values
  out_index(1) = -1_c_int64_t
  out_value(1) = -1.0_c_double

  ! Determine thread count and allocate per‑thread storage
  num_threads = omp_get_max_threads()
  allocate(thread_idx(num_threads))
  allocate(thread_val(num_threads))
  thread_idx = -1_c_int64_t
  thread_val = -1.0_c_double

  !$omp parallel private(tid, start, endi, i, local_idx, local_val)
    tid = omp_get_thread_num()
    start = (LEN_1D * tid) / num_threads + 1_c_int64_t
    endi = (LEN_1D * (tid + 1)) / num_threads
    local_idx = -1_c_int64_t
    local_val = -1.0_c_double
    do i = start, endi
      if (a(i) > k) then
        local_idx = i
        local_val = a(i)
        exit
      end if
    end do
    ! Store per‑thread result (Fortran arrays are 1‑based)
    thread_idx(tid+1) = local_idx
    thread_val(tid+1) = local_val
  !$omp end parallel

  ! Reduce to the smallest index found
  do i = 1, num_threads
    if (thread_idx(i) /= -1_c_int64_t) then
      if (out_index(1) == -1_c_int64_t .or. thread_idx(i) < out_index(1)) then
        out_index(1) = thread_idx(i)
        out_value(1) = thread_val(i)
      end if
    end if
  end do

end subroutine ext_break_capture_fp64
