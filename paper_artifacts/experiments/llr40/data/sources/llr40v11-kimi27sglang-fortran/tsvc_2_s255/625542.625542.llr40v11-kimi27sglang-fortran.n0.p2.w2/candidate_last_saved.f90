module tsvc_2_s255_m
  use iso_c_binding, only: c_double, c_int64_t
  implicit none
contains
  subroutine tsvc_2_s255_fp64(a, b, LEN_1D) bind(c, name="tsvc_2_s255_fp64")
    real(c_double), intent(inout) :: a(*)
    real(c_double), intent(in) :: b(*)
    integer(c_int64_t), value, intent(in) :: LEN_1D
    integer(c_int64_t) :: i, n
    real(c_double), parameter :: s = 0.333d0

    n = LEN_1D
    if (n <= 0_c_int64_t) return

    if (n == 1_c_int64_t) then
      ! numpy wraps b[-1] and b[-2] to the single element for length 1
      a(1) = (b(1) + b(1) + b(1)) * s
      return
    end if

    if (n == 2_c_int64_t) then
      a(1) = (b(1) + b(2) + b(1)) * s
      a(2) = (b(2) + b(1) + b(2)) * s
      return
    end if

    ! The scalar recurrence is equivalent to a 3-point wrapped moving sum.
    a(1) = (b(1) + b(n) + b(n - 1_c_int64_t)) * s
    a(2) = (b(2) + b(1) + b(n)) * s

    !$omp simd
    do i = 3_c_int64_t, n
      a(i) = ((b(i) + b(i - 1_c_int64_t)) + b(i - 2_c_int64_t)) * s
    end do
    !$omp end simd
  end subroutine tsvc_2_s255_fp64
end module tsvc_2_s255_m
