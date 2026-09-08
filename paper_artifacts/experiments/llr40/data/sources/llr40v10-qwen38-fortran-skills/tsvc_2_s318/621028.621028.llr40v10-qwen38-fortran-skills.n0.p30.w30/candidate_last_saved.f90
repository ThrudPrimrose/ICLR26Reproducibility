subroutine tsvc_2_s318_fp64(a, result, len_1d, inc) bind(C, name="tsvc_2_s318_fp64")
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: len_1d, inc
  real(c_double), intent(in) :: a(len_1d)
  real(c_double), intent(out) :: result(1)

  integer(c_int64_t) :: n, i, lo, hi, p, w, W
  integer(c_int64_t) :: g, ws
  integer :: nt, t, G
  real(c_double) :: mm, m, neginf
  real(c_double), allocatable :: smax(:), partm(:)
  integer(c_int64_t), allocatable :: parti(:)

  n = len_1d
  neginf = -huge(1.0d0)
  if (n < 1) then
    result(1) = 0.0d0
    return
  end if

  if (inc == 1) then
    nt = max(1, omp_get_max_threads())
    G = 16
    W = int(nt, 8) * G
    allocate(smax(W), partm(8 * nt), parti(8 * nt))

    ! pass 1: SIMD max over every sub-chunk (G sub-chunks per thread region)
    !$omp parallel do schedule(static, G)
    do w = 0, W - 1
      lo = (n * w) / W + 1
      hi = (n * (w + 1)) / W
      if (lo > hi) then
        smax(w + 1) = neginf
        cycle
      end if
      mm = neginf
      !$omp simd reduction(max:mm)
      do i = lo, hi
        mm = max(mm, abs(a(i)))
      end do
      smax(w + 1) = mm
    end do

    ! pass 2: per region, find the first sub-chunk holding the region max,
    ! then the first element inside it
    !$omp parallel do
    do t = 1, nt
      mm = neginf
      ws = (t - 1) * G + 1
      do g = 1, G
        if (smax(ws + g - 1) > mm) then
          mm = smax(ws + g - 1)
          ws = ws + g - 1
        end if
      end do
      partm(8 * t) = mm
      lo = (n * (ws - 1)) / W + 1
      hi = (n * ws) / W
      p = 0
      if (lo <= hi) then
        p = findloc(abs(a(lo:hi)), mm, dim=1)
      end if
      if (p > 0) then
        parti(8 * t) = lo + p - 2
      else
        parti(8 * t) = 0
      end if
    end do

    m = partm(8)
    do t = 2, nt
      if (partm(8 * t) > m) then
        m = partm(8 * t)
      end if
    end do
    p = 0
    do t = 1, nt
      if (partm(8 * t) == m) then
        p = parti(8 * t)
        exit
      end if
    end do
    result(1) = m + real(p, c_double)
    deallocate(smax, partm, parti)
  else
    ! general strided path (benchmark fixes inc = 1)
    mm = neginf
    !$omp parallel do reduction(max:mm)
    do i = 1, n
      mm = max(mm, abs(a(1 + (i - 1) * inc)))
    end do
    p = 0
    do i = 1, n
      if (p == 0 .and. abs(a(1 + (i - 1) * inc)) == mm) p = i
    end do
    result(1) = mm + real(p - 1, c_double)
  end if
end subroutine
