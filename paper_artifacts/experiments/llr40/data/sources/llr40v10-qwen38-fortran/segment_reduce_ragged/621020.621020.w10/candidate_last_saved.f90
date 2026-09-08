subroutine segment_reduce_ragged_fp64(row_ptr, val, w, out, nseg) bind(c, name="segment_reduce_ragged_fp64")
  use iso_c_binding
  implicit none
  integer, parameter :: i64 = c_int64_t
  integer(c_int64_t), value, intent(in) :: nseg
  integer(c_int64_t), intent(in) :: row_ptr(*)
  real(c_double), intent(in) :: val(*)
  real(c_double), intent(in) :: w(*)
  real(c_double), intent(out) :: out(*)
  integer(c_int64_t) :: s, e
  real(c_double) :: acc
  character(len=4096), target :: buf
  integer(c_int) :: pos, n

  interface
    function c_write(fd, p, n) bind(c, name='write')
      import
      integer(c_int) :: c_write
      integer(c_int), value :: fd
      type(c_ptr), value :: p
      integer(c_size_t), value :: n
    end function
  end interface

  buf = ' '
  pos = 1
  pos = put_int(pos, 'nseg=', nseg)
  pos = put_int(pos, ' rp1=', row_ptr(1))
  pos = put_int(pos, ' rp2=', row_ptr(2))
  pos = put_int(pos, ' rp3=', row_ptr(3))
  pos = put_int(pos, ' rp4=', row_ptr(4))
  pos = put_int(pos, ' rp_n=', row_ptr(nseg))
  pos = put_int(pos, ' rp_np1=', row_ptr(nseg+1))
  pos = put_hexd(pos, ' val1=', val(1))
  pos = put_hexd(pos, ' w1=', w(1))
  pos = put_hexd(pos, ' out1_in=', out(1))
  do s = 1, 4
    acc = 0.0d0
    do e = row_ptr(s)+1, row_ptr(s+1)
      acc = acc + val(e)*w(e)
    end do
    pos = put_int(pos, ' Csum_s', s)
    pos = put_hexd(pos, '=', acc)
  end do
  do s = 1, 4
    acc = 0.0d0
    do e = row_ptr(s), row_ptr(s+1)-1
      acc = acc + val(e)*w(e)
    end do
    pos = put_int(pos, ' Fsum_s', s)
    pos = put_hexd(pos, '=', acc)
  end do
  pos = put_c(pos, 10)
  n = c_write(1, c_loc(buf), int(pos-1, c_size_t))
  if (n < 1) then
    n = c_write(2, c_loc(buf), int(pos-1, c_size_t))
  end if

contains
  function put_int(pos, tag, v) result(np)
    integer(c_int), intent(inout) :: pos
    character(len=*), intent(in) :: tag
    integer(c_int64_t), intent(in) :: v
    integer(c_int) :: np
    integer(c_int64_t) :: t
    integer :: i, d
    integer :: tl
    character(len=64) :: tmp
    tl = len(tag)
    do i = 1, tl
      buf(pos:pos) = tag(i:i)
      pos = pos + 1
    end do
    t = v
    if (t < 0) then
      buf(pos:pos) = '-'; pos = pos + 1
      t = -t
    end if
    tmp = '                                                                '
    i = 64
    if (t == 0) then
      tmp(i:i) = '0'; i = i - 1
    end if
    do while (t > 0)
      d = int(mod(t, 10_i64))
      tmp(i:i) = char(48 + d)
      i = i - 1
      t = t / 10_i64
    end do
    do i = i+1, 64
      buf(pos:pos) = tmp(i:i)
      pos = pos + 1
    end do
    np = pos
  end function
  function put_hexd(pos, tag, v) result(np)
    integer(c_int), intent(inout) :: pos
    character(len=*), intent(in) :: tag
    real(c_double), intent(in) :: v
    integer(c_int) :: np
    integer(c_int64_t) :: b
    integer :: i, d
    integer :: tl
    character(len=1) :: c
    tl = len(tag)
    do i = 1, tl
      buf(pos:pos) = tag(i:i)
      pos = pos + 1
    end do
    b = transfer(v, 0_i64)
    do i = 0, 15
      d = int(mod(b, 16_i64))
      if (d < 10) then
        c = char(48 + d)
      else
        c = char(97 + d - 10)
      end if
      buf(pos:pos) = c
      pos = pos + 1
      b = b / 16_i64
    end do
    np = pos
  end function
  function put_c(pos, v) result(np)
    integer(c_int), intent(inout) :: pos
    integer, intent(in) :: v
    integer(c_int) :: np
    buf(pos:pos) = char(v)
    np = pos + 1
  end function
end subroutine segment_reduce_ragged_fp64
