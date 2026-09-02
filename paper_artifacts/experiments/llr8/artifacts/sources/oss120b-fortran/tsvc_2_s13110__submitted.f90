subroutine tsvc_2_s13110_fp64(aa, bb, len_2d, workspace, workspace_bytes) bind(C, name="tsvc_2_s13110_fp64")
  use iso_c_binding
  type(c_ptr), value :: aa
  type(c_ptr), value :: bb
  integer(c_int64_t), value :: len_2d
  type(c_ptr), value :: workspace
  integer(c_int64_t), value :: workspace_bytes

  real(c_double), pointer :: a(:)
  real(c_double), pointer :: b(:,:)
  integer(c_int64_t) :: n, idx, loc_idx
  real(c_double) :: maxv, thread_max, chksum
  integer(c_int64_t) :: thread_loc
  integer(c_int64_t) :: i0, j0

  ! Map C pointers to Fortran array pointers
  n = len_2d * len_2d
  call c_f_pointer(aa, a, [n])
  call c_f_pointer(bb, b, [2, 2])

  ! Initialize global max and location (zero‑based index)
  maxv = -huge(0.0_c_double)
  loc_idx = n   ! larger than any valid index

  !$omp parallel private(idx, thread_max, thread_loc) shared(maxv, loc_idx, a, n)
    thread_max = -huge(0.0_c_double)
    thread_loc = n
    !$omp do schedule(static)
    do idx = 1, n
      if (a(idx) > thread_max) then
        thread_max = a(idx)
        thread_loc = idx - 1_c_int64_t
      else if (a(idx) == thread_max) then
        ! Keep smallest index for equal values within this thread
        if (idx - 1_c_int64_t < thread_loc) thread_loc = idx - 1_c_int64_t
      end if
    end do
    !$omp critical
      if (thread_max > maxv) then
        maxv = thread_max
        loc_idx = thread_loc
      else if (thread_max == maxv) then
        if (thread_loc < loc_idx) loc_idx = thread_loc
      end if
    !$omp end critical
  !$omp end parallel

  ! Convert flat index to 2‑D zero‑based coordinates (row‑major order)
  i0 = loc_idx / len_2d
  j0 = mod(loc_idx, len_2d)

  ! Compute checksum using zero‑based indices
  chksum = maxv + real(i0, kind=c_double) + real(j0, kind=c_double)
  b(1,1) = chksum
end subroutine tsvc_2_s13110_fp64
