subroutine tsvc_2_s3112_fp64(a, b, LEN_1D) bind(c, name='tsvc_2_s3112_fp64')
  use, intrinsic :: iso_c_binding
  use omp_lib
  implicit none
  integer(C_INT64_T), value :: LEN_1D
  real(C_DOUBLE), dimension(LEN_1D), intent(in) :: a
  real(C_DOUBLE), dimension(LEN_1D), intent(inout) :: b
  integer(C_INT64_T) :: i, j, tid, nthreads, start, end, istart
  real(C_DOUBLE) :: sum, offset, total
  real(C_DOUBLE), allocatable :: offsets(:)
  integer(C_INT64_T) :: chunk_size
  integer, parameter :: VL = 16
  real(C_DOUBLE) :: v(VL)
  
  nthreads = omp_get_max_threads()
  chunk_size = (LEN_1D + nthreads - 1_C_INT64_T) / nthreads
  allocate(offsets(nthreads))
  
  !$omp parallel private(tid, start, end, i, j, istart, sum, offset, v)
  tid = omp_get_thread_num()
  start = tid * chunk_size + 1_C_INT64_T
  end = min(start + chunk_size - 1_C_INT64_T, LEN_1D)
  
  if (start <= end) then
    ! Pass 1: compute chunk sum
    offsets(tid + 1) = sum(a(start:end))
  else
    offsets(tid + 1) = 0.0_C_DOUBLE
  end if
  !$omp barrier
  
  !$omp single
  total = 0.0_C_DOUBLE
  do i = 1, nthreads
    offsets(i) = offsets(i) + total
    total = offsets(i)
  end do
  !$omp end single
  
  if (start <= end) then
    ! Pass 2: compute prefix sums with offset
    if (tid > 0) then
      offset = offsets(tid)
    else
      offset = 0.0_C_DOUBLE
    end if
    sum = offset
    
    istart = start
    do while (istart <= end .and. mod(istart - 1_C_INT64_T, int(VL, C_INT64_T)) /= 0)
      sum = sum + a(istart)
      b(istart) = sum
      istart = istart + 1_C_INT64_T
    end do
    
    do i = istart, end - VL + 1_C_INT64_T, VL
      v(1:VL) = a(i:i+VL-1_C_INT64_T)
      v(2:VL) = v(2:VL) + v(1:VL-1)
      v(3:VL) = v(3:VL) + v(1:VL-2)
      v(5:VL) = v(5:VL) + v(1:VL-4)
      v(9:VL) = v(9:VL) + v(1:VL-8)
      b(i:i+VL-1_C_INT64_T) = v(1:VL) + sum
      sum = sum + v(VL)
    end do
    
    do i = i, end
      sum = sum + a(i)
      b(i) = sum
    end do
  end if
  !$omp end parallel
  
  deallocate(offsets)
end subroutine tsvc_2_s3112_fp64
