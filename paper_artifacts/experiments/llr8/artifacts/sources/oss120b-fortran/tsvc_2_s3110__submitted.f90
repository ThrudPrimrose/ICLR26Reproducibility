! Fortran implementation of TSVC kernel tsvc_2_s3110 for double precision.
! Identifies the maximum element in a 2D array, records its zero-based indices,
! computes a checksum and stores it in bb(1,1).
!
! Expected C interface (as per the harness):
!   void tsvc_2_s3110_fp64(double *, double *, int64_t, uint8_t *, int64_t);
!
! The Fortran subroutine uses BIND(C) to provide the exact symbol name and
! matches argument types via iso_c_binding.

subroutine tsvc_2_s3110_fp64(aa, bb, LEN_2D, workspace, workspace_bytes) bind(C, name="tsvc_2_s3110_fp64")
  use iso_c_binding
  implicit none
  ! Arguments from C side
  integer(C_INT64_T), value :: LEN_2D
  type(C_PTR), value :: workspace          ! unused workspace pointer
  integer(C_INT64_T), value :: workspace_bytes ! size of workspace in bytes
  real(C_DOUBLE), intent(in) :: aa(LEN_2D, LEN_2D)
  real(C_DOUBLE), intent(inout) :: bb(2,2)

  integer :: idx(2)
  real(C_DOUBLE) :: maxv
  integer :: xindex, yindex
  real(C_DOUBLE) :: chksum
  integer :: i, j
  real(C_DOUBLE) :: local_max
  integer :: local_x, local_y

  ! Compute maximum value and its location using an OpenMP parallel reduction.
  ! Initialize global max and indices.
  maxv = aa(1,1)
  idx = [1, 1]
  ! Parallel region: each thread computes a local max and its indices.
  !$omp parallel default(shared) private(i, j, local_max, local_x, local_y)
    local_max = -huge(0.0d0)
    local_x = -1
    local_y = -1
    !$omp do nowait schedule(static)
    do i = 1, LEN_2D
      do j = 1, LEN_2D
        if (aa(i, j) > local_max) then
          local_max = aa(i, j)
          local_x = i - 1
          local_y = j - 1
        end if
      end do
    end do
    !$omp critical
    if (local_max > maxv) then
      maxv = local_max
      idx(1) = local_x + 1  ! store as 1‑based temporary for later conversion
      idx(2) = local_y + 1
    end if
    !$omp end critical
  !$omp end parallel
  ! Convert from stored 1‑based indices to zero‑based.
  xindex = idx(1) - 1
  yindex = idx(2) - 1

  chksum = maxv + real(xindex, kind=C_DOUBLE) + real(yindex, kind=C_DOUBLE)
  bb(1,1) = chksum

  ! The rest of bb is untouched; workspace is ignored.
end subroutine tsvc_2_s3110_fp64
