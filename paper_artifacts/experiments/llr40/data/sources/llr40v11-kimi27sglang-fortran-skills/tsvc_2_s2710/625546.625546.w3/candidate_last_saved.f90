subroutine tsvc_2_s2710_fp64(a, b, c, d, e, x, n) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: n
  real(c_double), intent(inout) :: a(n), b(n), c(n)
  real(c_double), intent(in) :: d(n), e(n), x(n)
  integer(c_int64_t) :: i

  if (n <= 0) return

  if (x(1) > 0.0d0) then
    if (n > 10) then
      !$omp parallel do simd schedule(static) if(n > 4096)
      do i = 1, n
        if (a(i) > b(i)) then
          a(i) = a(i) + b(i) * d(i)
          c(i) = c(i) + d(i) * d(i)
        else
          b(i) = a(i) + e(i) * e(i)
          c(i) = a(i) + d(i) * d(i)
        end if
      end do
    else
      !$omp parallel do simd schedule(static) if(n > 4096)
      do i = 1, n
        if (a(i) > b(i)) then
          a(i) = a(i) + b(i) * d(i)
          c(i) = d(i) * e(i) + 1.0d0
        else
          b(i) = a(i) + e(i) * e(i)
          c(i) = a(i) + d(i) * d(i)
        end if
      end do
    end if
  else
    if (n > 10) then
      !$omp parallel do simd schedule(static) if(n > 4096)
      do i = 1, n
        if (a(i) > b(i)) then
          a(i) = a(i) + b(i) * d(i)
          c(i) = c(i) + d(i) * d(i)
        else
          b(i) = a(i) + e(i) * e(i)
          c(i) = c(i) + e(i) * e(i)
        end if
      end do
    else
      !$omp parallel do simd schedule(static) if(n > 4096)
      do i = 1, n
        if (a(i) > b(i)) then
          a(i) = a(i) + b(i) * d(i)
          c(i) = d(i) * e(i) + 1.0d0
        else
          b(i) = a(i) + e(i) * e(i)
          c(i) = c(i) + e(i) * e(i)
        end if
      end do
    end if
  end if
end subroutine tsvc_2_s2710_fp64
