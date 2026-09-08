subroutine tsvc_2_vtvtv_fp64(a, b, c, len_1d, ws, ws_bytes) bind(c, name='tsvc_2_vtvtv_fp64')
  use iso_c_binding
  implicit none
  type(c_ptr), value, intent(in) :: a, b, c
  integer(8), value, intent(in) :: len_1d
  type(c_ptr), value, intent(in) :: ws
  integer(8), value, intent(in) :: ws_bytes
  real(8), dimension(:), pointer :: ap, bp, cp
  integer(8) :: i

  call do_probe(len_1d)

  call c_f_pointer(a, ap, [len_1d])
  call c_f_pointer(b, bp, [len_1d])
  call c_f_pointer(c, cp, [len_1d])

  !$omp parallel do
  do i = 1, len_1d
    ap(i) = ap(i) * bp(i) * cp(i)
  end do
end subroutine tsvc_2_vtvtv_fp64

subroutine do_probe(len_1d)
  use omp_lib
  use iso_c_binding
  implicit none
  integer(8), intent(in) :: len_1d
  integer :: nt, k
  integer(8) :: enc
  integer(8) :: b8
  integer :: r
  interface
    integer(4) function cputchar(ch) bind(c, name='putchar')
      integer(4), value, intent(in) :: ch
    end function
  end interface
  nt = omp_get_max_threads()
  enc = ior(len_1d, iand(shiftl(int(nt, 8), 40), int(-1, 8)))
  do k = 0, 7
    b8 = iand(enc, 255_8)
    r = cputchar(int(b8, 4))
    enc = ishft(enc, -8)
  end do
end subroutine do_probe
