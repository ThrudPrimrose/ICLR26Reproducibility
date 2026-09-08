subroutine tsvc_2_s2233_fp64(aa_c, bb_c, cc_c, LEN_2D) bind(C, name='tsvc_2_s2233_fp64')
  use iso_c_binding
  implicit none
  type(c_ptr), intent(in)    :: aa_c, bb_c, cc_c
  integer(c_int64_t), value  :: LEN_2D
  real(c_double), dimension(:,:), pointer :: aa, bb, cc
  integer(c_int32_t) :: n, a, b
  n = int(LEN_2D, c_int32_t)
  call c_f_pointer(aa_c, aa, [n, n])
  call c_f_pointer(bb_c, bb, [n, n])
  call c_f_pointer(cc_c, cc, [n, n])
  ! aa_C[r][c] = aa_F[c+1, r+1].  C loops r,c in 8..N-1 (0-based) => Fortran 9..N (1-based).
  ! Both loops reduce to: OUT(a,b) = OUT(a,b-1) + cc(a,b) for a,b in 9..N (a=col, b=row).
  do a = 9, n
     do b = 9, n
        aa(a, b) = aa(a, b-1) + cc(a, b)
     end do
     do b = 9, n
        bb(a, b) = bb(a, b-1) + cc(a, b)
     end do
  end do
end subroutine tsvc_2_s2233_fp64
