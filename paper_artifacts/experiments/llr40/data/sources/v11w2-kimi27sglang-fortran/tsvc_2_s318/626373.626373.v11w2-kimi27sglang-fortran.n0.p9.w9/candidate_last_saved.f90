module tsvc_2_s318_m
  use iso_c_binding, only: c_double, c_int64_t
  use omp_lib
  implicit none
  integer(c_int64_t), parameter :: MAXTHR = 128_c_int64_t
contains
  subroutine tsvc_2_s318_fp64(a, result, LEN_1D, inc) bind(c, name="tsvc_2_s318_fp64")
    real(c_double), intent(in) :: a(*)
    real(c_double), intent(out) :: result
    integer(c_int64_t), value, intent(in) :: LEN_1D, inc
    integer(c_int64_t) :: i, k, idx, nthreads, nthr, tid, start, endd, chunk, t
    real(c_double) :: maxv
    real(c_double) :: lmax(0_c_int64_t:MAXTHR - 1_c_int64_t)

    if (LEN_1D <= 0_c_int64_t) then
      result = 0.0_c_double
      return
    end if

    nthreads = int(omp_get_max_threads(), c_int64_t)
    if (nthreads > MAXTHR) nthreads = MAXTHR

    if (LEN_1D <= 4096_c_int64_t .or. nthreads <= 1_c_int64_t) then
      maxv = abs(a(1))
      idx = 0_c_int64_t
      k = inc
      do i = 1_c_int64_t, LEN_1D - 1_c_int64_t
        if (abs(a(1_c_int64_t + k)) > maxv) then
          maxv = abs(a(1_c_int64_t + k))
          idx = i
        end if
        k = k + inc
      end do
      result = maxv + real(idx, c_double)
      return
    end if

    lmax = 0.0_c_double

    if (inc == 1_c_int64_t) then
      !$omp parallel private(tid, nthr, start, endd, chunk, i, maxv) shared(lmax) num_threads(int(nthreads))
      tid = int(omp_get_thread_num(), c_int64_t)
      nthr = int(omp_get_num_threads(), c_int64_t)
      chunk = (LEN_1D + nthr - 1_c_int64_t) / nthr
      start = tid * chunk
      endd = min(start + chunk, LEN_1D) - 1_c_int64_t
      maxv = 0.0_c_double
      !$omp simd reduction(max:maxv)
      do i = start + 1_c_int64_t, endd + 1_c_int64_t
        maxv = max(maxv, abs(a(i)))
      end do
      !$omp end simd
      lmax(tid) = maxv
      !$omp end parallel
    else
      !$omp parallel private(tid, nthr, start, endd, chunk, i, maxv) shared(lmax) num_threads(int(nthreads))
      tid = int(omp_get_thread_num(), c_int64_t)
      nthr = int(omp_get_num_threads(), c_int64_t)
      chunk = (LEN_1D + nthr - 1_c_int64_t) / nthr
      start = tid * chunk
      endd = min(start + chunk, LEN_1D) - 1_c_int64_t
      maxv = 0.0_c_double
      do i = start, endd
        maxv = max(maxv, abs(a(1_c_int64_t + i * inc)))
      end do
      lmax(tid) = maxv
      !$omp end parallel
    end if

    maxv = lmax(0_c_int64_t)
    do t = 1_c_int64_t, nthreads - 1_c_int64_t
      if (lmax(t) > maxv) maxv = lmax(t)
    end do

    idx = 0_c_int64_t
    k = 0_c_int64_t
    do i = 0_c_int64_t, LEN_1D - 1_c_int64_t
      if (abs(a(1_c_int64_t + k)) >= maxv) then
        idx = i
        exit
      end if
      k = k + inc
    end do

    result = maxv + real(idx, c_double)
  end subroutine tsvc_2_s318_fp64
end module tsvc_2_s318_m
