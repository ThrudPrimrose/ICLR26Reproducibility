module kernel_m
  use, intrinsic :: iso_c_binding
  use omp_lib
  implicit none
contains
  subroutine tsvc_2_s231_fp64(aa, bb, LEN_2D) bind(c, name='tsvc_2_s231_fp64')
    integer(c_int64_t), value :: LEN_2D
    real(c_double), intent(inout) :: aa(LEN_2D, LEN_2D)
    real(c_double), intent(in) :: bb(LEN_2D, LEN_2D)
    integer :: i, j, n, tid, nthreads, chunk, i0, i1
    n = int(LEN_2D)
    if (n <= 512) then
      do j = 2, n
        do i = 1, n
          aa(i, j) = aa(i, j-1) + bb(i, j)
        end do
      end do
      return
    end if
    !$omp parallel private(i, j, tid, nthreads, chunk, i0, i1)
    tid = omp_get_thread_num()
    nthreads = omp_get_num_threads()
    chunk = (n + nthreads - 1) / nthreads
    i0 = tid * chunk + 1
    i1 = min((tid + 1) * chunk, n)
    if (i0 <= n) then
      do j = 2, n
        !$omp simd
        do i = i0, i1
          aa(i, j) = aa(i, j-1) + bb(i, j)
        end do
      end do
    end if
    !$omp end parallel
  end subroutine tsvc_2_s231_fp64
end module kernel_m
