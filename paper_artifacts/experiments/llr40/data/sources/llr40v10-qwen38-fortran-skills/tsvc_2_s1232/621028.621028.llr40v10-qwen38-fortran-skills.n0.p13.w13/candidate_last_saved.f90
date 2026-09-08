subroutine tsvc_2_s1232_fp64(aa, bb, cc, len_2d, vlen) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: len_2d, vlen
  real(c_double), intent(inout) :: aa(len_2d, len_2d)
  real(c_double), intent(in) :: bb(len_2d, len_2d), cc(len_2d, len_2d)

  integer(c_int64_t) :: i, j, njj
  integer(c_int64_t) :: vl

  vl = max(vlen, 1_c_int64_t)

  ! numpy: for j: for i in j*vlen..N-1: aa[i,j] = bb[i,j]+cc[i,j]
  ! C layout aa[i*N+j]  <=>  Fortran aa(j, i).
  ! Row-oriented rewrite: for each i (row), columns j = 1..(i-1)/vlen+1
  ! are touched -- unit stride in the FIRST subscript.
  !
  !$omp parallel do simd schedule(static)
  do i = 1, len_2d
    njj = (i - 1) / vl + 1
    do j = 1, njj
      aa(j, i) = bb(j, i) + cc(j, i)
    end do
  end do
end subroutine tsvc_2_s1232_fp64
