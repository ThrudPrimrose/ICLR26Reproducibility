! TSVC tsvc_2/s231: aa(j,i) = aa(j-1,i) + bb(j,i)  (j = 1..N-1, i = 0..N-1), C row-major.
! Column-wise prefix sums; each column independent -> parallel over columns.
! ABI note: this gfortran passes bind(C) dummies by reference except assumed-size
! arrays, so arrays are taken as aa(*)/bb(*) and the size via c_loc(n) (raw register).
subroutine tsvc_2_s231_fp64(aa, bb, len_d) bind(C, name="tsvc_2_s231_fp64")
  use, intrinsic :: iso_c_binding
  implicit none
  real(c_double), intent(inout) :: aa(*)
  real(c_double), intent(in)    :: bb(*)
  integer(c_int64_t), target, intent(in) :: len_d(*)
  type(c_ptr) :: cpn
  integer(c_int64_t) :: n64
  integer :: N, i, j
  real(c_double) :: acc

  cpn = c_loc(len_d)
  n64 = transfer(cpn, 0_c_int64_t)
  N = int(n64)
  if (N < 2) return

  !$omp parallel do default(none) shared(aa, bb, N) private(i, j, acc)
  do i = 1, N
    acc = aa(i)
    do j = 1, N-1
      acc = acc + bb(j*N + i)
      aa(j*N + i) = acc
    end do
  end do
end subroutine tsvc_2_s231_fp64
