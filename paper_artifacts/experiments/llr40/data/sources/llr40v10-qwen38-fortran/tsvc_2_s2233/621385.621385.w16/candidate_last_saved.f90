subroutine tsvc_2_s2233_fp64(aa, bb, cc, n) bind(C, name='tsvc_2_s2233_fp64')
  use, intrinsic :: iso_c_binding
  implicit none
  real(c_double), intent(inout) :: aa(*)
  real(c_double), intent(inout) :: bb(*)
  real(c_double), intent(in)    :: cc(*)
  integer(c_int64_t), value :: n

  integer, parameter :: ACCMAX = 8000000
  real(c_double), save :: acc(ACCMAX)

  interface
    subroutine tsvc2_scan16f(arr, cc, n, i0)
      use, intrinsic :: iso_c_binding
      real(c_double), intent(inout) :: arr(*)
      real(c_double), intent(in)    :: cc(*)
      integer(c_int64_t), value :: n, i0
    end subroutine tsvc2_scan16f
  end interface

  integer(c_int64_t) :: i, j, base

  if (n > 8) then
    if (n - 8 <= ACCMAX) then
      do i = 1, n - 8
        acc(i) = aa(7*n + 9 + (i - 1))
      end do
      do j = 8, n - 1
        base = j*n + 9
        do i = 0, n - 9
          acc(i + 1) = acc(i + 1) + cc(base + i)
          aa(base + i) = acc(i + 1)
        end do
      end do

      do i = 1, n - 8
        acc(i) = bb(7*n + 9 + (i - 1))
      end do
      do j = 8, n - 1
        base = j*n + 9
        do i = 0, n - 9
          acc(i + 1) = acc(i + 1) + cc(base + i)
          bb(base + i) = acc(i + 1)
        end do
      end do
    else
      ! oversized fallback: 16-wide register chunks, serial
      do i = 8, n - 1
        call tsvc2_scan16f(aa, cc, n, i)
        call tsvc2_scan16f(bb, cc, n, i)
      end do
    end if
  end if
end subroutine tsvc_2_s2233_fp64

subroutine tsvc2_scan16f(arr, cc, n, i0)
  use, intrinsic :: iso_c_binding
  implicit none
  real(c_double), intent(inout) :: arr(*)
  real(c_double), intent(in)    :: cc(*)
  integer(c_int64_t), value :: n, i0
  integer(c_int64_t) :: j, m
  real(c_double) :: acc(0:15)

  do m = 0, 15
    acc(m) = arr(7*n + i0 + 1 + m)
  end do
  do j = 8, n - 1
    do m = 0, 15
      acc(m) = acc(m) + cc(j*n + i0 + 1 + m)
    end do
    do m = 0, 15
      arr(j*n + i0 + 1 + m) = acc(m)
    end do
  end do
end subroutine tsvc2_scan16f
