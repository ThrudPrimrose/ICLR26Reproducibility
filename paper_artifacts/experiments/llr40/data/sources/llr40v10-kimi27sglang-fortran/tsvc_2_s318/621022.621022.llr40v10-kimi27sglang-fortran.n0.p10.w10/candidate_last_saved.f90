module tsvc_2_s318_mod
  use iso_c_binding
  use omp_lib
  implicit none

  type :: maxloc_t
    real(c_double) :: maxv
    integer(c_int64_t) :: idx
  end type maxloc_t

contains

  pure function maxloc_combine(a, b) result(res)
    type(maxloc_t), intent(in) :: a, b
    type(maxloc_t) :: res
    if (a%maxv > b%maxv) then
      res = a
    else if (b%maxv > a%maxv) then
      res = b
    else
      if (a%idx < b%idx) then
        res = a
      else
        res = b
      end if
    end if
  end function maxloc_combine

  pure function maxloc_init() result(res)
    type(maxloc_t) :: res
    res%maxv = -huge(1.0_c_double)
    res%idx = huge(1_c_int64_t)
  end function maxloc_init

  subroutine tsvc_2_s318_fp64(a, result, LEN_1D, inc) bind(c, name='tsvc_2_s318_fp64')
    real(c_double), intent(in) :: a(*)
    real(c_double), intent(out) :: result(*)
    integer(c_int64_t), value, intent(in) :: LEN_1D, inc

    integer(c_int64_t) :: i
    integer :: nthreads, chunk
    real(c_double) :: v
    logical :: mask
    type(maxloc_t) :: best

    !$omp declare reduction(maxloc : maxloc_t : omp_out = maxloc_combine(omp_out, omp_in)) initializer(omp_priv = maxloc_init())

    best = maxloc_t(abs(a(1)), 0_c_int64_t)

    nthreads = omp_get_max_threads()
    chunk = int(max(256_c_int64_t, min(4096_c_int64_t, LEN_1D / int(nthreads, c_int64_t) / 16_c_int64_t)))
    call omp_set_schedule(omp_sched_guided, chunk)

    !$omp parallel do reduction(maxloc:best) schedule(runtime) if(LEN_1D > 4096_c_int64_t)
    do i = 2_c_int64_t, LEN_1D
      v = abs(a(1_c_int64_t + inc * (i - 1_c_int64_t)))
      mask = v > best%maxv
      best%maxv = merge(v, best%maxv, mask)
      best%idx = merge(i - 1_c_int64_t, best%idx, mask)
    end do
    !$omp end parallel do

    result(1) = best%maxv + dble(best%idx)
  end subroutine tsvc_2_s318_fp64
end module tsvc_2_s318_mod
