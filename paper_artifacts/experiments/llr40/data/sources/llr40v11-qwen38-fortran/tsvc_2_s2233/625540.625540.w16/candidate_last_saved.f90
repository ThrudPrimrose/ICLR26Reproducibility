subroutine tsvc_2_s2233_fp64(aa, bb, cc, LEN_2D) bind(C, name='tsvc_2_s2233_fp64')
  use iso_c_binding
  implicit none
  real(c_double), intent(inout), dimension(*) :: aa
  real(c_double), intent(inout), dimension(*) :: bb
  real(c_double), intent(in),    dimension(*) :: cc
  integer(c_int64_t), value :: LEN_2D

  integer(c_int64_t) :: L, k, i
  L = LEN_2D
  do k = 8, L-1
     !$omp simd
     do i = 9, L
        aa(k*L+i) = aa((k-1)*L+i) + cc(k*L+i)
     end do
     !$omp simd
     do i = 9, L
        bb(k*L+i) = bb((k-1)*L+i) + cc(k*L+i)
     end do
  end do
end subroutine
