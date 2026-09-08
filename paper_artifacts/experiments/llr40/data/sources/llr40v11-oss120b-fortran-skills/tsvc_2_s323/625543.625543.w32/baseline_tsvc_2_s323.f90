subroutine tsvc_2_s323_fp64(a, b, c, d, e, LEN_1D) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D
  real(c_double), intent(inout) :: a(LEN_1D)
  real(c_double), intent(inout) :: b(LEN_1D)
  real(c_double), intent(in) :: c(LEN_1D)
  real(c_double), intent(in) :: d(LEN_1D)
  real(c_double), intent(in) :: e(LEN_1D)
  integer(c_int64_t) :: i
  real(c_double) :: term_cd, term_ce
  real(c_double) :: prefix_cd, prefix_ce

  ! First pass: compute inclusive prefix of c*d and store in a(i).
  prefix_cd = 0.0d0
  !$omp parallel do reduction(inscan, +: prefix_cd) private(term_cd)
  do i = 2, LEN_1D
    term_cd = c(i) * d(i)
    prefix_cd = prefix_cd + term_cd
    !$omp scan inclusive(prefix_cd)
    a(i) = prefix_cd
  end do

  ! Second pass: compute inclusive prefix of c*e, combine with the stored c*d prefix
  ! to obtain final a(i) and b(i).
  prefix_ce = 0.0d0
  !$omp parallel do reduction(inscan, +: prefix_ce) private(term_ce)
  do i = 2, LEN_1D
    term_ce = c(i) * e(i)
    prefix_ce = prefix_ce + term_ce
    !$omp scan inclusive(prefix_ce)
    ! b(i) = b(1) + sum_{k=2..i} c(k)*d(k) + sum_{k=2..i} c(k)*e(k)
    b(i) = b(1) + a(i) + prefix_ce
    ! a(i) = b(i) - c(i)*e(i) (equivalently a(i) = b(1) + sum_{k=2..i} c(k)*d(k) + sum_{k=2..i-1} c(k)*e(k))
    a(i) = b(i) - term_ce
  end do

end subroutine tsvc_2_s323_fp64
