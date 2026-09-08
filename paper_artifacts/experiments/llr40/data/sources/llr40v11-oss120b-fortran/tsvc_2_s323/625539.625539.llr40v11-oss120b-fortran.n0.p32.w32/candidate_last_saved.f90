module tsvc_2_s323_mod
  use iso_c_binding
  implicit none
contains
  subroutine tsvc_2_s323_fp64(a, b, c, d, e, LEN_1D) bind(C, name="tsvc_2_s323_fp64")
    ! Compute products in parallel, then sequentially apply recurrence to preserve exact rounding.
    real(c_double), intent(inout) :: a(*)
    real(c_double), intent(inout) :: b(*)
    real(c_double), intent(in) :: c(*)
    real(c_double), intent(in) :: d(*)
    real(c_double), intent(in) :: e(*)
    integer(c_int64_t), value :: LEN_1D
    integer(c_int64_t) :: i, N
    real(c_double), allocatable :: cd(:), ce(:)
    real(c_double) :: prev_b, b_val
    N = LEN_1D
    if (N <= 1_c_int64_t) return
    allocate(cd(2:N))
    allocate(ce(2:N))
    ! Compute c*d in parallel (vectorizable)
    !$omp parallel do private(i) schedule(static)
    do i = 2, N
      cd(i) = c(i) * d(i)
    end do
    !$omp end parallel do
    ! Compute c*e in parallel
    !$omp parallel do private(i) schedule(static)
    do i = 2, N
      ce(i) = c(i) * e(i)
    end do
    !$omp end parallel do
    ! Apply recurrence sequentially to preserve exact rounding order
    ! New sequential recurrence using register-cached previous b value
    prev_b = b(1)
    do i = 2, N
      a(i) = prev_b + cd(i)
      b_val = a(i) + ce(i)
      b(i) = b_val
      prev_b = b_val
    end do
    deallocate(cd, ce)
  end subroutine tsvc_2_s323_fp64
end module tsvc_2_s323_mod
