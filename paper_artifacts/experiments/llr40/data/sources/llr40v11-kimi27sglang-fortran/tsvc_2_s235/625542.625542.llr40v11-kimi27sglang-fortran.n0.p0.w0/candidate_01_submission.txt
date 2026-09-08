module tsvc_2_s235_mod
  use, intrinsic :: iso_c_binding, only: c_double, c_int64_t
  implicit none
contains
  subroutine tsvc_2_s235_fp64(a, aa, b, bb, c, LEN_2D) bind(C, name='tsvc_2_s235_fp64')
    use omp_lib
    integer(c_int64_t), value, intent(in) :: LEN_2D
    real(c_double), intent(inout) :: a(LEN_2D)
    real(c_double), intent(inout) :: aa(LEN_2D, LEN_2D)
    real(c_double), intent(in) :: b(LEN_2D), c(LEN_2D), bb(LEN_2D, LEN_2D)
    integer(c_int64_t) :: i, j, n
    integer :: tid, nt
    integer(c_int64_t) :: istart, iend

    n = LEN_2D

    !$omp parallel private(i, j, tid, nt, istart, iend)
    tid = omp_get_thread_num()
    nt = omp_get_num_threads()
    istart = (int(n, kind=c_int64_t) * tid) / nt + 1
    iend = (int(n, kind=c_int64_t) * (tid + 1)) / nt

    do i = istart, iend
       a(i) = a(i) + b(i) * c(i)
    end do

    do j = 2, n
       !$omp simd
       do i = istart, iend
          aa(i, j) = aa(i, j - 1) + bb(i, j) * a(i)
       end do
    end do
    !$omp end parallel
  end subroutine tsvc_2_s235_fp64
end module tsvc_2_s235_mod
