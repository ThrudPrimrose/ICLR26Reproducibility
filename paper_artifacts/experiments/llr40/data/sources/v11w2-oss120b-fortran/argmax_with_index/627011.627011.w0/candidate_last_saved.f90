subroutine argmax_with_index_fp64(a, out_index, out_value, len_1d) bind(C, name="argmax_with_index_fp64")
  use iso_c_binding
  use omp_lib
  implicit none
  real(c_double), intent(in) :: a(*)
  integer(c_int64_t), intent(out) :: out_index(*)
  real(c_double), intent(out) :: out_value(*)
  integer(c_int64_t), value :: len_1d

  integer(c_int64_t) :: i, tid, nthreads
  real(c_double) :: thread_max, max_val
  integer(c_int64_t) :: thread_idx, max_idx
  real(c_double), allocatable :: max_vals(:)
  integer(c_int64_t), allocatable :: max_idxs(:)

  if (len_1d <= 0_c_int64_t) then
    out_value(1) = 0.0_c_double
    out_index(1) = -1_c_int64_t
    return
  end if

  nthreads = omp_get_max_threads()
  allocate(max_vals(nthreads))
  allocate(max_idxs(nthreads))

  ! Initialize per-thread temporary storage
  do i = 1, nthreads
    max_vals(i) = a(1)
    max_idxs(i) = 0_c_int64_t
  end do

  !$omp parallel private(tid, i, thread_max, thread_idx)
    tid = omp_get_thread_num() + 1  ! Fortran 1-based indexing
    thread_max = a(1)
    thread_idx = 0_c_int64_t
    !$omp do schedule(static)
    do i = 1, len_1d
      if (a(i) > thread_max) then
        thread_max = a(i)
        thread_idx = i - 1_c_int64_t
      end if
    end do
    !$omp end do
    max_vals(tid) = thread_max
    max_idxs(tid) = thread_idx
  !$omp end parallel

  ! Reduce across threads to find global max and first index
  max_val = -huge(0.0_c_double)
  max_idx = -1_c_int64_t
  do i = 1, nthreads
    if (max_vals(i) > max_val) then
      max_val = max_vals(i)
      max_idx = max_idxs(i)
    else if (max_vals(i) == max_val) then
      if (max_idxs(i) >= 0_c_int64_t .and. max_idxs(i) < max_idx) then
        max_idx = max_idxs(i)
      end if
    end if
  end do

  ! If no element was processed (should not happen for len_1d>0), fallback
  if (max_idx < 0_c_int64_t) then
    max_idx = 0_c_int64_t
    max_val = a(1)
  end if

  out_value(1) = max_val
  out_index(1) = max_idx
end subroutine argmax_with_index_fp64
