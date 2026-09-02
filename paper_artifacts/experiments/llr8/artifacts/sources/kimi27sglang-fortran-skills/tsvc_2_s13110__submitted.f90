subroutine tsvc_2_s13110_fp64(aa, bb, LEN_2D) bind(C)
  use iso_c_binding
  use omp_lib
  implicit none

  type :: maxidx_t
    real(c_double) :: val
    integer(c_int64_t) :: idx
  end type maxidx_t

  !$omp declare reduction(maxidx : maxidx_t : &
  !$omp& omp_out = merge(omp_in, omp_out, &
  !$omp&   omp_in%val > omp_out%val .or. &
  !$omp&   (omp_in%val == omp_out%val .and. omp_in%idx < omp_out%idx))) &
  !$omp& initializer(omp_priv = maxidx_t(-huge(1.0d0), huge(1_c_int64_t)))

  integer(c_int64_t), value, intent(in) :: LEN_2D
  real(c_double), intent(in) :: aa(LEN_2D, LEN_2D)
  real(c_double), intent(inout) :: bb(2, 2)

  integer(c_int64_t) :: i, j, xindex, yindex
  real(c_double) :: chksum
  type(maxidx_t) :: mi

  mi = maxidx_t(aa(1, 1), 0_c_int64_t)

  !$omp parallel do simd reduction(maxidx:mi) private(j)
  do i = 1, LEN_2D
    do j = 1, LEN_2D
      if (aa(j, i) > mi%val) then
        mi = maxidx_t(aa(j, i), (i - 1) * LEN_2D + (j - 1))
      end if
    end do
  end do
  !$omp end parallel do simd

  xindex = mi%idx / LEN_2D
  yindex = mod(mi%idx, LEN_2D)
  chksum = mi%val + dble(xindex) + dble(yindex)
  bb(1, 1) = chksum
end subroutine tsvc_2_s13110_fp64
