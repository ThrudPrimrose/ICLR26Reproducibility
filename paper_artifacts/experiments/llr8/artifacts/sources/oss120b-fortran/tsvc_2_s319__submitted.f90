! Fortran implementation of TSVC kernel s319
! The kernel computes:
!   a[i] = c[i] + d[i]
!   b[i] = c[i] + e[i]
!   sum_val accumulates a[i] and b[i] over the loop
!   b[0] = sum_val (the final total sum)
!
! The reference Python implementation is:
!   def s319(a, b, c, d, e, LEN_1D):
!       sum_val = 0.0
!       for i in range(LEN_1D):
!           a[i] = c[i] + d[i]
!           sum_val = sum_val + a[i]
!           b[i] = c[i] + e[i]
!           sum_val = sum_val + b[i]
!       b[0] = sum_val
!
! This subroutine follows the C-ABI expected by the benchmark harness. It is declared with BIND(C) and uses ISO_C_BINDING types.

module tsvc_2_s319_mod
  use iso_c_binding, only: c_int, c_int64_t, c_double, c_ptr
  implicit none
contains
  subroutine tsvc_2_s319_fp64(a, b, c, d, e, len_1d, workspace, workspace_bytes) bind(C, name="tsvc_2_s319_fp64")
    real(c_double), intent(inout) :: a(*)
    real(c_double), intent(inout) :: b(*)
    real(c_double), intent(in) :: c(*)
    real(c_double), intent(in) :: d(*)
    real(c_double), intent(in) :: e(*)
    integer(c_int64_t), value :: len_1d
    type(c_ptr), value :: workspace
    integer(c_int64_t), value :: workspace_bytes
    integer(c_int64_t) :: i
    real(c_double) :: sum_val
    sum_val = 0.0_c_double
!$omp parallel do default(none) shared(a,b,c,d,e,len_1d) private(i) reduction(+:sum_val)
    do i = 1, len_1d
      a(i) = c(i) + d(i)
      sum_val = sum_val + a(i)
      b(i) = c(i) + e(i)
      sum_val = sum_val + b(i)
    end do
!$omp end parallel do
    b(1) = sum_val
  end subroutine tsvc_2_s319_fp64
end module tsvc_2_s319_mod
