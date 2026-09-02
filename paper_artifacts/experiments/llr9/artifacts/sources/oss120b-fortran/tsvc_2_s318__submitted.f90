module tsvc_2_s318_mod
  use iso_c_binding
  implicit none
contains

subroutine tsvc_2_s318_fp64(a, result, LEN_1D, inc, workspace, workspace_len) bind(C, name="tsvc_2_s318_fp64")
  ! Fortran implementation of the TSVC kernel s318 (double precision).
  ! The driver expects the C symbol tsvc_2_s318_fp64 with the following signature:
  !   void tsvc_2_s318_fp64(double *, double *, int64_t, int64_t, uint8_t *, int64_t);
  ! The workspace arguments are ignored.
  implicit none

  ! Arguments from the C side.
  real(c_double), intent(in) :: a(*)                ! input array
  real(c_double), intent(out) :: result(1)          ! output scalar (as length‑1 array)
  integer(c_int64_t), value :: LEN_1D, inc         ! length and stride
  type(c_ptr), value :: workspace                  ! unused workspace pointer
  integer(c_int64_t), value :: workspace_len      ! unused workspace size

  integer :: i, idx
  integer :: inc_i, N
  real(c_double) :: maxv, v

  ! Convert the 64‑bit integers to default integers for looping.
  inc_i = int(inc)
  N = int(LEN_1D)

  maxv = -1.0_c_double

  ! Compute the maximum absolute value using a parallel reduction.
!$omp parallel do schedule(static) reduction(max:maxv) private(v) shared(a, inc_i, N)
  do i = 0, N - 1
    v = abs(a(1 + i * inc_i))
    if (v > maxv) maxv = v
  end do
!$omp end parallel do

  ! Find the first index where the maximum occurs.
  idx = -1
  do i = 0, N - 1
    if (abs(a(1 + i * inc_i)) == maxv) then
      idx = i
      exit
    end if
  end do
  if (idx < 0) idx = 0

  result(1) = maxv + dble(idx)
end subroutine tsvc_2_s318_fp64

end module tsvc_2_s318_mod
