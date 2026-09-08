subroutine tsvc_2_s119_fp64(aa, bb, n) bind(C, name='tsvc_2_s119_fp64')
  use iso_c_binding
  implicit none
  type(c_ptr), value :: aa
  type(c_ptr), value, intent(in) :: bb
  integer(c_int64_t), value :: n
  interface
    integer function c_write(fd, buf, count) bind(C, name='write')
      import
      integer, value :: fd
      type(c_ptr), value :: buf
      integer, value :: count
    end function c_write
  end interface
  character(len=64), target :: line
  integer :: iv
  write(line, '(A,I16)') 'LEN_2D=', n
  iv = c_write(1, c_loc(line), 23)
end subroutine
