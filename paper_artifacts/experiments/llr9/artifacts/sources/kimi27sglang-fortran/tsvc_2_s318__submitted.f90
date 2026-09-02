subroutine tsvc_2_s318_fp64(a, result, LEN_1D, inc) bind(c, name="tsvc_2_s318_fp64")
  use iso_c_binding
  use omp_lib
  implicit none
  real(c_double), intent(in), target :: a(*)
  real(c_double), intent(out) :: result(*)
  integer(c_int64_t), intent(in), value :: LEN_1D, inc

  integer(c_int64_t) :: i, k, idx, nthreads, tid, istart, iend, j
  real(c_double) :: maxv, v
  real(c_double), allocatable :: tmaxv(:)
  integer(c_int64_t), allocatable :: tidx(:)
  type(c_ptr) :: aptr
  real(c_double), pointer :: pa(:)

  if (LEN_1D <= 0) then
    result(1) = 0.0_c_double
    return
  end if

  if (inc == 1) then
    aptr = c_loc(a(1))
    call c_f_pointer(aptr, pa, [LEN_1D])

    if (LEN_1D < 10000) then
      maxv = abs(pa(1))
      idx = 0
      do i = 2, LEN_1D
        v = abs(pa(i))
        if (v > maxv) then
          maxv = v
          idx = i - 1
        end if
      end do
      result(1) = maxv + real(idx, c_double)
      return
    end if

    nthreads = omp_get_max_threads()
    allocate(tmaxv(nthreads), tidx(nthreads))

    !$omp parallel private(tid, istart, iend, maxv, idx, i, v)
    tid = omp_get_thread_num()
    istart = tid * LEN_1D / nthreads
    iend = (tid + 1) * LEN_1D / nthreads

    idx = istart
    maxv = abs(pa(istart + 1))
    do i = istart + 1, iend - 1
      v = abs(pa(i + 1))
      if (v > maxv) then
        maxv = v
        idx = i
      end if
    end do

    tmaxv(tid + 1) = maxv
    tidx(tid + 1) = idx
    !$omp end parallel

    maxv = tmaxv(1)
    idx = tidx(1)
    do j = 2, nthreads
      if (tmaxv(j) > maxv) then
        maxv = tmaxv(j)
        idx = tidx(j)
      end if
    end do

    result(1) = maxv + real(idx, c_double)
    return
  end if

  ! Strided path (inc /= 1)
  if (LEN_1D < 10000) then
    maxv = abs(a(1))
    idx = 0
    k = inc
    do i = 1, LEN_1D - 1
      v = abs(a(k + 1))
      if (v > maxv) then
        maxv = v
        idx = i
      end if
      k = k + inc
    end do
    result(1) = maxv + real(idx, c_double)
    return
  end if

  nthreads = omp_get_max_threads()
  allocate(tmaxv(nthreads), tidx(nthreads))

  !$omp parallel private(tid, istart, iend, maxv, idx, i, k, v)
  tid = omp_get_thread_num()
  istart = tid * LEN_1D / nthreads
  iend = (tid + 1) * LEN_1D / nthreads

  idx = istart
  k = istart * inc
  maxv = abs(a(k + 1))
  k = k + inc

  do i = istart + 1, iend - 1
    v = abs(a(k + 1))
    if (v > maxv) then
      maxv = v
      idx = i
    end if
    k = k + inc
  end do

  tmaxv(tid + 1) = maxv
  tidx(tid + 1) = idx
  !$omp end parallel

  maxv = tmaxv(1)
  idx = tidx(1)
  do j = 2, nthreads
    if (tmaxv(j) > maxv) then
      maxv = tmaxv(j)
      idx = tidx(j)
    end if
  end do

  result(1) = maxv + real(idx, c_double)
end subroutine tsvc_2_s318_fp64
