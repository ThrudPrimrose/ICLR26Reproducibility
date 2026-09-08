module tsvc_2_s255_mod
  use iso_c_binding, only: c_double, c_int64_t
  implicit none
  integer(c_int64_t), parameter :: W = 8_c_int64_t
contains
  subroutine tsvc_2_s255_fp64(a, b, LEN_1D) bind(C,name="tsvc_2_s255_fp64")
    real(c_double), intent(inout) :: a(*)
    real(c_double), intent(in)    :: b(*)
    integer(c_int64_t), value, intent(in) :: LEN_1D
    integer(c_int64_t) :: i, n, m, mW, iend
    real(c_double) :: x, y
    real(c_double) :: t(W + 2)

    n = LEN_1D
    if (n <= 0) return
    if (n == 1) then
      x = b(1)
      a(1) = (b(1) + x + x) * 0.333_c_double
      return
    end if

    x = b(n)
    y = b(n - 1)
    do i = 1, 2
      a(i) = (b(i) + x + y) * 0.333_c_double
      y = x
      x = b(i)
    end do

    if (n >= 3) then
      m = n - 2
      mW = (m / W) * W
      iend = 2 + mW
      do i = 3, iend, W
        t(1:W+2) = b(i-2:i+W-1)
        a(i:i+W-1) = (t(3:W+2) + t(2:W+1) + t(1:W)) * 0.333_c_double
      end do
      do i = iend + 1, n
        a(i) = (b(i) + b(i - 1) + b(i - 2)) * 0.333_c_double
      end do
    end if
  end subroutine tsvc_2_s255_fp64
end module tsvc_2_s255_mod
