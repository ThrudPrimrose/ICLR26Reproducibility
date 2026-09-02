subroutine tsvc_2_s233_fp64(aa, bb, cc, len_2d, pws, ws_bytes) bind(C, name="tsvc_2_s233_fp64")
  use, intrinsic :: iso_c_binding
  use omp_lib
  implicit none
  real(c_double) :: aa(*), bb(*), cc(*)
  integer(c_int64_t), value, intent(in) :: len_2d
  type(c_ptr), value, intent(in) :: pws
  integer(c_int64_t), value, intent(in) :: ws_bytes

  integer :: N, C, R, t, nthr, per, rem, total, c0, c1, ro, rp

  N = int(len_2d)
  if (N > 8) then
    ! Row-major (C) layout: numpy(j,i) 0-based = aa(j*N + i + 1).
    ! Fortran 1-based (C=col+1, R=row+1): numpy(R-1,C-1) = aa((R-1)*N + C).
    ! ONE parallel region for both parts (they touch disjoint arrays) to cut
    ! OpenMP spawn/barrier overhead.
    !$omp parallel private(t, nthr, per, rem, total, c0, c1, R, C, ro, rp)
      ! ---- Part 1 (aa): A(C,R)=A(C,R-1)+CC(C,R); dependence in R (front), C parallel.
      ! Wavefront: each thread owns a contiguous column-chunk, steps R in lockstep, SIMD.
      nthr  = omp_get_num_threads()
      total = N - 8
      per   = total / nthr
      rem   = total - per*nthr
      t     = omp_get_thread_num()
      c0    = 9 + t*per + min(t, rem)
      if (t < rem) then
        c1 = c0 + per
      else
        c1 = c0 + per - 1
      end if
      do R = 9, N
        ro = (R - 1) * N
        rp = (R - 2) * N
        !$omp simd
        do C = c0, c1
          aa(ro + C) = aa(rp + C) + cc(ro + C)
        end do
      end do

      ! ---- Part 2 (bb): B(C,R)=B(C-1,R)+CC(C,R); dependence in C (serial), R parallel.
      ! Row-parallel: each row is a serial unit-stride prefix sum over C.
      !$omp do schedule(static)
      do R = 9, N
        ro = (R - 1) * N
        do C = 9, N
          bb(ro + C) = bb(ro + C - 1) + cc(ro + C)
        end do
      end do
      !$omp end do
    !$omp end parallel
  end if
end subroutine tsvc_2_s233_fp64
