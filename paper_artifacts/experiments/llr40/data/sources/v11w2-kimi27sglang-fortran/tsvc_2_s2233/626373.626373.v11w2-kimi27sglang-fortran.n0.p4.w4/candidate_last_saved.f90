subroutine tsvc_2_s2233_fp64(aa, bb, cc, LEN_2D) bind(C, name="tsvc_2_s2233_fp64")
  use iso_c_binding, only: c_double, c_int64_t
  implicit none
  integer(c_int64_t), value :: LEN_2D
  real(c_double), intent(inout) :: aa(*)
  real(c_double), intent(inout) :: bb(*)
  real(c_double), intent(in) :: cc(*)
  integer(c_int64_t) :: i, r, idp, idc, n

  n = LEN_2D
  !$omp parallel private(r, i, idp, idc)
  do r = 9, n
     !$omp do simd schedule(static)
     do i = 9, n
        idp = (r - 2) * n + i
        idc = idp + n
        aa(idc) = aa(idp) + cc(idc)
        bb(idc) = bb(idp) + cc(idc)
     end do
     !$omp end do simd
  end do
  !$omp end parallel
end subroutine tsvc_2_s2233_fp64
