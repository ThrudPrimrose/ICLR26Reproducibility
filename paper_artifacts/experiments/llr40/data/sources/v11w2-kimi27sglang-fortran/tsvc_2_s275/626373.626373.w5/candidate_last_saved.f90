module tsvc_2_s275_mod
  use iso_c_binding, only : c_int64_t, c_double
  implicit none
contains
  subroutine tsvc_2_s275_fp64(aa, bb, cc, n) bind(c, name='tsvc_2_s275_fp64')
    integer(c_int64_t), value     :: n
    real(c_double), intent(inout) :: aa(n,n)
    real(c_double), intent(in)    :: bb(n,n)
    real(c_double), intent(in)    :: cc(n,n)
    logical :: mask(n)
    integer(c_int64_t) :: i, j, ib, ie
    integer(c_int64_t), parameter :: block = 512_c_int64_t
    do i = 1, n
      mask(i) = aa(i,1) > 0.0_c_double
    end do
    !$omp parallel do schedule(dynamic,1) private(j,i,ie) default(shared)
    do ib = 1, n, block
      ie = min(ib + block - 1, n)
      do j = 2, n
        do i = ib, ie
          aa(i,j) = merge(aa(i,j-1) + bb(i,j) * cc(i,j), aa(i,j), mask(i))
        end do
      end do
    end do
    !$omp end parallel do
  end subroutine tsvc_2_s275_fp64
end module tsvc_2_s275_mod
