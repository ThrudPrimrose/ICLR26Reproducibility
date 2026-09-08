subroutine tsvc_2_s3110_fp64(aa, bb, LEN_2D) bind(C, name="tsvc_2_s3110_fp64")
  use iso_c_binding
  implicit none
  real(c_double), intent(in)       :: aa(*)
  real(c_double), intent(out)      :: bb(*)
  integer(c_int64_t), intent(in), value :: LEN_2D

  integer(c_int64_t) :: N, CH, nchunks
  integer(c_int64_t) :: k, c, lo, hi
  real(c_double) :: maxv, chksum, m
  real(c_double), allocatable :: chunkmax(:)
  integer(c_int64_t) :: cstar, k0, i0, j0

  N = LEN_2D * LEN_2D
  if (N < 1) then
     bb(1) = 0.0d0
     return
  end if
  CH = 4096_c_int64_t
  nchunks = (N + CH - 1_c_int64_t) / CH
  allocate(chunkmax(nchunks))

  !$omp parallel do num_threads(16) schedule(static)
  do c = 1_c_int64_t, nchunks
     lo = (c-1_c_int64_t)*CH + 1_c_int64_t
     hi = min(c*CH, N)
     m = aa(lo)
     do k = lo+1_c_int64_t, hi
        m = max(m, aa(k))
     end do
     chunkmax(c) = m
  end do
  !$omp end parallel do

  maxv = chunkmax(1_c_int64_t)
  cstar = 1_c_int64_t
  do c = 2_c_int64_t, nchunks
     if (chunkmax(c) > maxv) then
        maxv = chunkmax(c)
        cstar = c
     end if
  end do

  lo = (cstar-1_c_int64_t)*CH + 1_c_int64_t
  hi = min(cstar*CH, N)
  k0 = lo - 1_c_int64_t
  do k = lo, hi
     if (aa(k) == maxv) then
        k0 = k - 1_c_int64_t
        exit
     end if
  end do

  i0 = k0 / LEN_2D
  j0 = k0 - i0*LEN_2D
  chksum = maxv + dble(i0) + dble(j0)
  bb(1) = chksum
end subroutine tsvc_2_s3110_fp64
