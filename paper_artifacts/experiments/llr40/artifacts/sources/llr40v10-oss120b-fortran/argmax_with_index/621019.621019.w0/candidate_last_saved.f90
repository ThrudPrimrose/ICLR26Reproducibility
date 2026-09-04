module argmax_with_index_mod
  use iso_c_binding, only: c_double, c_int64_t, c_int
  use omp_lib
  implicit none
contains
  subroutine argmax_with_index_fp64(a, out_index, out_value, LEN_1D) bind(C, name="argmax_with_index_fp64")
    implicit none
    ! Arguments
    real(c_double), intent(in) :: a(*)
    integer(c_int64_t), intent(out) :: out_index(1)
    real(c_double), intent(out) :: out_value(1)
    integer(c_int64_t), value :: LEN_1D
    
    ! Local variables
    integer(c_int64_t) :: i
    integer(c_int) :: tid, nthreads
    real(c_double) :: local_max_val, global_max_val
    integer(c_int64_t) :: local_max_idx, global_max_idx
    integer(c_int64_t), allocatable :: thread_idx(:)
    real(c_double), allocatable :: thread_val(:)
    
    if (LEN_1D <= 0) then
      ! Undefined behavior; just set to zero
      out_value(1) = 0.0_c_double
      out_index(1) = 0_c_int64_t
      return
    end if
    
    ! Number of threads
    nthreads = omp_get_max_threads()
    allocate(thread_idx(nthreads))
    allocate(thread_val(nthreads))
    ! Initialize arrays to sentinel values
    thread_val = -huge(1.0_c_double)
    thread_idx = 0_c_int64_t
    
    !$omp parallel private(i, tid, local_max_val, local_max_idx, nthreads) default(shared)
      tid = omp_get_thread_num()
      nthreads = omp_get_num_threads()
      ! Initialise each thread's local maximum with the first element
      local_max_val = a(1)
      local_max_idx = 1_c_int64_t

      i = tid + 1   ! Fortran 1-indexed loop start
      do while (i <= LEN_1D)
        if (a(i) > local_max_val) then
          local_max_val = a(i)
          local_max_idx = i
        end if
        i = i + nthreads
      end do

      thread_val(tid+1) = local_max_val
      thread_idx(tid+1) = local_max_idx
    !$omp end parallel
    
    ! Reduce across threads (sequential reduction)
    global_max_val = thread_val(1)
    global_max_idx = thread_idx(1)
    do i = 2, nthreads
      if (thread_val(i) > global_max_val) then
        global_max_val = thread_val(i)
        global_max_idx = thread_idx(i)
      else if (thread_val(i) == global_max_val) then
        if (thread_idx(i) < global_max_idx) then
          global_max_idx = thread_idx(i)
        end if
      end if
    end do
    
    out_value(1) = global_max_val
    out_index(1) = global_max_idx - 1_c_int64_t   ! Convert to 0-based index for C callers
    
    deallocate(thread_idx)
    deallocate(thread_val)
  end subroutine argmax_with_index_fp64

end module argmax_with_index_mod
