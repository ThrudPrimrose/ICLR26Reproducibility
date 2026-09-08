subroutine tsvc_2_s1232_fp64(aa, bb, cc, LEN_2D, VLEN) bind(C, name='tsvc_2_s1232_fp64')
  use iso_c_binding
  use omp_lib
  implicit none
  type(c_ptr), value :: aa, bb, cc
  integer(c_int64_t), value :: LEN_2D, VLEN
  real(c_double), dimension(:,:), pointer :: a, b, c
  integer(c_int64_t) :: ii, jmax
  call c_f_pointer(aa, a, [LEN_2D, LEN_2D])
  call c_f_pointer(bb, b, [LEN_2D, LEN_2D])
  call c_f_pointer(cc, c, [LEN_2D, LEN_2D])
  call omp_set_num_threads(16)
  if (VLEN <= 0) then
    a = b + c
    return
  end if
  !$omp parallel do default(none) shared(a,b,c,LEN_2D,VLEN) private(ii,jmax) schedule(dynamic,16)
  do ii = 1, LEN_2D
    jmax = (ii-1)/VLEN
    if (jmax > LEN_2D-1) jmax = LEN_2D-1
    a(1:jmax+1, ii) = b(1:jmax+1, ii) + c(1:jmax+1, ii)
  end do
end subroutine
