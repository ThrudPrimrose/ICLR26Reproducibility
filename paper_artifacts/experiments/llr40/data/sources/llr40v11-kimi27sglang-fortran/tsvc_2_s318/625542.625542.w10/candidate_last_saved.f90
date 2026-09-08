module tsvc_2_s318_mod
  use, intrinsic :: iso_c_binding
  use omp_lib
  implicit none
contains
  subroutine tsvc_2_s318_fp64(a, result, LEN_1D, inc) bind(c, name="tsvc_2_s318_fp64")
    real(c_double), dimension(*), intent(in) :: a
    real(c_double), dimension(*), intent(out) :: result
    integer(c_int64_t), value :: LEN_1D, inc
    integer(c_int64_t) :: nt, t, tid, start, finish, i, k
    integer(c_int64_t) :: index, lidx
    real(c_double) :: maxv, v, lmaxv
    real(c_double), allocatable :: thread_maxv(:)
    integer(c_int64_t), allocatable :: thread_idx(:)
    integer(c_int64_t) :: loc

    ! Very small inputs: let the compiler/library vectorize the reduction.
    if (LEN_1D <= 4096_c_int64_t) then
      if (inc == 1_c_int64_t) then
        loc = maxloc(abs(a(1_c_int64_t:LEN_1D)), dim=1, kind=c_int64_t)
        index = loc - 1_c_int64_t
        maxv = abs(a(loc))
      else
        k = 0_c_int64_t
        index = 0_c_int64_t
        maxv = abs(a(1_c_int64_t))
        do i = 1_c_int64_t, LEN_1D - 1_c_int64_t
          k = k + inc
          v = abs(a(k + 1_c_int64_t))
          if (v > maxv) then
            index = i
            maxv = v
          end if
        end do
      end if
      result(1) = maxv + real(index, c_double)
      return
    end if

    ! Medium inputs: simple scalar loop avoids parallel-region overhead.
    if (LEN_1D <= 100000_c_int64_t) then
      k = 0_c_int64_t
      index = 0_c_int64_t
      maxv = abs(a(1_c_int64_t))
      do i = 1_c_int64_t, LEN_1D - 1_c_int64_t
        k = k + inc
        v = abs(a(k + 1_c_int64_t))
        if (v > maxv) then
          index = i
          maxv = v
        end if
      end do
      result(1) = maxv + real(index, c_double)
      return
    end if

    nt = int(omp_get_max_threads(), c_int64_t)
    if (LEN_1D < nt) nt = LEN_1D

    allocate(thread_maxv(nt), thread_idx(nt))

    !$omp parallel do schedule(static,1) private(tid,start,finish,i,k,lmaxv,lidx,v) shared(thread_maxv,thread_idx) num_threads(int(nt, kind=kind(0)))
    do t = 1_c_int64_t, nt
      tid = int(omp_get_thread_num(), c_int64_t) + 1_c_int64_t
      start = (t - 1_c_int64_t) * (LEN_1D / nt) + min(t - 1_c_int64_t, mod(LEN_1D, nt))
      finish = start + (LEN_1D / nt) + merge(1_c_int64_t, 0_c_int64_t, t - 1_c_int64_t < mod(LEN_1D, nt))

      k = start * inc
      lmaxv = abs(a(k + 1_c_int64_t))
      lidx = start
      do i = start + 1_c_int64_t, finish - 1_c_int64_t
        k = k + inc
        v = abs(a(k + 1_c_int64_t))
        if (v > lmaxv) then
          lidx = i
          lmaxv = v
        end if
      end do
      thread_maxv(tid) = lmaxv
      thread_idx(tid) = lidx
    end do

    maxv = thread_maxv(1_c_int64_t)
    index = thread_idx(1_c_int64_t)
    do t = 2_c_int64_t, nt
      if (thread_maxv(t) > maxv) then
        maxv = thread_maxv(t)
        index = thread_idx(t)
      end if
    end do

    result(1) = maxv + real(index, c_double)
    deallocate(thread_maxv, thread_idx)
  end subroutine
end module
