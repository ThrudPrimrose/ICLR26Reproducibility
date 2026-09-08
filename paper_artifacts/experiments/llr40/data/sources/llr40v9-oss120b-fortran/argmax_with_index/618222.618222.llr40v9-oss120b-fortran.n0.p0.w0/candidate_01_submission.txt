module argmax_mod
  use iso_c_binding
  use omp_lib
  implicit none
contains
  subroutine argmax_with_index_fp64(a, out_index, out_value, LEN_1D) bind(C, name="argmax_with_index_fp64")
    ! Arguments: a (input array of double), out_index (output int64), out_value (output double), LEN_1D (size)
    real(c_double), intent(in) :: a(*)
    integer(c_int64_t), intent(out) :: out_index(*)
    real(c_double), intent(out) :: out_value(*)
    integer(c_int64_t), value :: LEN_1D
    ! Local variables
    real(c_double) :: max_val
    integer(c_int64_t) :: max_idx
    integer :: nthreads, tid
    integer(c_int64_t) :: i
    real(c_double), allocatable :: local_max(:)
    integer(c_int64_t), allocatable :: local_idx(:)
    real(c_double) :: thread_max
    integer(c_int64_t) :: thread_idx

    if (LEN_1D <= 0) then
      out_value(1) = 0.0_c_double
      out_index(1) = 0_c_int64_t
      return
    end if

    ! Determine number of OpenMP threads
    nthreads = omp_get_max_threads()
    allocate(local_max(nthreads))
    allocate(local_idx(nthreads))
    ! Initialize per-thread maxima to very low value
    do i = 1, nthreads
       local_max(i) = -huge(0.0_c_double)
       local_idx(i) = 0_c_int64_t
    end do

    !$omp parallel private(tid, i, thread_max, thread_idx)
    tid = omp_get_thread_num() + 1  ! Fortran arrays are 1-indexed
    thread_max = -huge(0.0_c_double)
    thread_idx = 0_c_int64_t
    !$omp do schedule(static) nowait
    do i = 1, LEN_1D
       if (a(i) > thread_max) then
          thread_max = a(i)
          thread_idx = i
       end if
    end do
    !$omp end do
    local_max(tid) = thread_max
    local_idx(tid) = thread_idx
    !$omp end parallel

    ! Reduce across threads to find global maximum and its index
    max_val = local_max(1)
    max_idx = local_idx(1)
    do i = 2, nthreads
       if (local_max(i) > max_val) then
          max_val = local_max(i)
          max_idx = local_idx(i)
       end if
    end do

    out_value(1) = max_val
    out_index(1) = max_idx

  end subroutine argmax_with_index_fp64
end module argmax_mod
