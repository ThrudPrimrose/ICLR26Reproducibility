subroutine tsvc_2_s2710_fp64(a, b, c, d, e, x, len_1d) bind(C, name="tsvc_2_s2710_fp64")
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: len_1d
  real(c_double), intent(inout) :: a(len_1d), b(len_1d), c(len_1d)
  real(c_double), intent(in) :: d(len_1d), e(len_1d), x(len_1d)
  integer(c_int64_t) :: i
  logical :: m

  if (len_1d > 10) then
    if (x(1) > 0.0d0) then
      ! c(i) = (a>b) ? c(i)+d(i)^2 : a(i)+d(i)^2
      !$omp parallel do simd default(none) shared(a,b,c,d,e,len_1d) private(m) schedule(static)
      do i = 1, len_1d
        m = a(i) > b(i)
        a(i) = merge(a(i) + b(i) * d(i), a(i), m)
        b(i) = merge(b(i), a(i) + e(i) * e(i), m)
        c(i) = merge(c(i) + d(i) * d(i), a(i) + d(i) * d(i), m)
      end do
    else
      ! c(i) = c(i) + (a>b) ? d(i)^2 : e(i)^2
      !$omp parallel do simd default(none) shared(a,b,c,d,e,len_1d) private(m) schedule(static)
      do i = 1, len_1d
        m = a(i) > b(i)
        a(i) = merge(a(i) + b(i) * d(i), a(i), m)
        b(i) = merge(b(i), a(i) + e(i) * e(i), m)
        c(i) = merge(c(i) + d(i) * d(i), c(i) + e(i) * e(i), m)
      end do
    end if
  else
    if (x(1) > 0.0d0) then
      ! c(i) = (a>b) ? d(i)*e(i)+1 : a(i)+d(i)^2
      !$omp parallel do simd default(none) shared(a,b,c,d,e,len_1d) private(m) schedule(static)
      do i = 1, len_1d
        m = a(i) > b(i)
        a(i) = merge(a(i) + b(i) * d(i), a(i), m)
        b(i) = merge(b(i), a(i) + e(i) * e(i), m)
        c(i) = merge(d(i) * e(i) + 1.0d0, a(i) + d(i) * d(i), m)
      end do
    else
      ! c(i) = (a>b) ? d(i)*e(i)+1 : c(i)+e(i)^2
      !$omp parallel do simd default(none) shared(a,b,c,d,e,len_1d) private(m) schedule(static)
      do i = 1, len_1d
        m = a(i) > b(i)
        a(i) = merge(a(i) + b(i) * d(i), a(i), m)
        b(i) = merge(b(i), a(i) + e(i) * e(i), m)
        c(i) = merge(d(i) * e(i) + 1.0d0, c(i) + e(i) * e(i), m)
      end do
    end if
  end if
end subroutine tsvc_2_s2710_fp64
