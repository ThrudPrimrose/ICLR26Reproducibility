module kernel
  use, intrinsic :: iso_c_binding, only: c_double, c_int64_t
  implicit none
contains
  subroutine tsvc_2_s2710_fp64(a, b, c, d, e, x, LEN_1D) bind(c, name='tsvc_2_s2710_fp64')
    integer(c_int64_t), value, intent(in) :: LEN_1D
    real(c_double), dimension(LEN_1D), intent(inout) :: a, b, c
    real(c_double), dimension(LEN_1D), intent(in) :: d, e, x
    integer(c_int64_t) :: i

    if (LEN_1D > 10_c_int64_t) then
      if (x(1) > 0.0_c_double) then
        !$omp parallel do simd schedule(static)
        do i = 1, LEN_1D
          if (a(i) > b(i)) then
            a(i) = a(i) + b(i) * d(i)
            c(i) = c(i) + d(i) * d(i)
          else
            b(i) = a(i) + e(i) * e(i)
            c(i) = a(i) + d(i) * d(i)
          end if
        end do
        !$omp end parallel do simd
      else
        !$omp parallel do simd schedule(static)
        do i = 1, LEN_1D
          if (a(i) > b(i)) then
            a(i) = a(i) + b(i) * d(i)
            c(i) = c(i) + d(i) * d(i)
          else
            b(i) = a(i) + e(i) * e(i)
            c(i) = c(i) + e(i) * e(i)
          end if
        end do
        !$omp end parallel do simd
      end if
    else
      if (x(1) > 0.0_c_double) then
        !$omp parallel do simd schedule(static)
        do i = 1, LEN_1D
          if (a(i) > b(i)) then
            a(i) = a(i) + b(i) * d(i)
            c(i) = d(i) * e(i) + 1.0_c_double
          else
            b(i) = a(i) + e(i) * e(i)
            c(i) = a(i) + d(i) * d(i)
          end if
        end do
        !$omp end parallel do simd
      else
        !$omp parallel do simd schedule(static)
        do i = 1, LEN_1D
          if (a(i) > b(i)) then
            a(i) = a(i) + b(i) * d(i)
            c(i) = d(i) * e(i) + 1.0_c_double
          else
            b(i) = a(i) + e(i) * e(i)
            c(i) = c(i) + e(i) * e(i)
          end if
        end do
        !$omp end parallel do simd
      end if
    end if
  end subroutine tsvc_2_s2710_fp64
end module kernel
