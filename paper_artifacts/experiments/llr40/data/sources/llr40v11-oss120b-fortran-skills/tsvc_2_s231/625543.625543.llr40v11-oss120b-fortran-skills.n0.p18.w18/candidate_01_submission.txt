subroutine tsvc_2_s231_fp64(a, b, len_2d) bind(C, name="tsvc_2_s231_fp64")
  use iso_c_binding
  use omp_lib
  implicit none
  real(c_double), intent(inout) :: a(*)
  real(c_double), intent(in) :: b(*)
  integer(c_int64_t), value :: len_2d
  integer(c_int64_t) :: i, j
  integer :: tid, nthreads, start_i, end_i
  !$omp parallel default(none) shared(a,b,len_2d) private(tid, nthreads, start_i, end_i, i, j)
    tid = omp_get_thread_num()
    nthreads = omp_get_num_threads()
    start_i = (len_2d * tid) / nthreads + 1
    end_i = (len_2d * (tid + 1)) / nthreads
    do j = 2, len_2d
      !$omp simd
      do i = start_i, end_i
        a((j-1)*len_2d + i) = a((j-2)*len_2d + i) + b((j-1)*len_2d + i)
      end do
    end do
  !$omp end parallel
end subroutine tsvc_2_s231_fp64
