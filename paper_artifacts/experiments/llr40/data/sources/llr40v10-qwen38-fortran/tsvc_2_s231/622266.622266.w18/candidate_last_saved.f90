subroutine tsvc_2_s231_fp64(aa_p, bb_p, len2d) bind(C, name='tsvc_2_s231_fp64')
  use iso_c_binding
  implicit none
  type(c_ptr), value, intent(in) :: aa_p, bb_p
  integer(c_int64_t), value, intent(in) :: len2d
  real(c_double), pointer :: aa(:), bb(:)
  integer(c_int64_t) :: n, c, e, r, nrow, nchunk, e0, ap, bp, cend
  integer(c_int64_t) :: rowp, strideb, s16
  real(c_double) :: s

  n = len2d
  if (n <= 1) return
  nrow = n - 1
  nchunk = (n + 7) / 8
  call c_f_pointer(aa_p, aa, [n*n])
  call c_f_pointer(bb_p, bb, [n*n])

  !$omp parallel
  !$omp do schedule(static) private(c,e,r,e0,s)
  do c = 0, nchunk-1
    e0 = 8*c
    do e = 1, 8
      if (e0 + e > n) exit
      s = aa(e0 + e)
      do r = 1, nrow
        s = s + bb(r*n + e0 + e)
        aa(r*n + e0 + e) = s
      end do
    end do
  end do
  !$omp end do
  !$omp end parallel
end subroutine tsvc_2_s231_fp64
