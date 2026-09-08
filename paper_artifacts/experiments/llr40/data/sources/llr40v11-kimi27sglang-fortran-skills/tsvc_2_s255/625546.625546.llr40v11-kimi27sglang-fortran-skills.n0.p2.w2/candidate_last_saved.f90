module tsvc_2_s255_mod
  use iso_c_binding, only: c_int64_t, c_double
  implicit none
contains

  subroutine tsvc_2_s255_fp64(a, b, n) bind(C, name='tsvc_2_s255_fp64')
    integer(c_int64_t), value, intent(in) :: n
    real(c_double), intent(inout) :: a(n)
    real(c_double), intent(in)    :: b(n)
    call tsvc_2_s255_impl(a, b, n)
  end subroutine tsvc_2_s255_fp64

  subroutine tsvc_2_s255(a, b, n) bind(C, name='tsvc_2_s255')
    integer(c_int64_t), value, intent(in) :: n
    real(c_double), intent(inout) :: a(n)
    real(c_double), intent(in)    :: b(n)
    call tsvc_2_s255_impl(a, b, n)
  end subroutine tsvc_2_s255

  subroutine tsvc_2_s255_impl(a, b, n)
    integer(c_int64_t), value, intent(in) :: n
    real(c_double), intent(inout) :: a(n)
    real(c_double), intent(in)    :: b(n)
    integer(c_int64_t) :: i
    real(c_double) :: x, y

    if (n < 3_c_int64_t) then
      if (n <= 0_c_int64_t) return
      x = b(n)
      if (n >= 2_c_int64_t) then
        y = b(n - 1_c_int64_t)
      else
        y = x
      end if
      do i = 1_c_int64_t, n
        a(i) = (b(i) + x + y) * 0.333d0
        y = x
        x = b(i)
      end do
      return
    end if

    a(1) = (b(1) + b(n) + b(n - 1_c_int64_t)) * 0.333d0
    a(2) = (b(2) + b(1) + b(n)) * 0.333d0

    !$omp parallel do simd schedule(static)
    do i = 3_c_int64_t, n
      a(i) = (b(i) + b(i - 1_c_int64_t) + b(i - 2_c_int64_t)) * 0.333d0
    end do
    !$omp end parallel do simd
  end subroutine tsvc_2_s255_impl

end module tsvc_2_s255_mod
