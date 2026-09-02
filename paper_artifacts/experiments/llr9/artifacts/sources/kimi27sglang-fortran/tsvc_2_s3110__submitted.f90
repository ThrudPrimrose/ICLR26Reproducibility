module pair_mod
  use iso_c_binding, only: c_double, c_int64_t
  implicit none

  type :: pair
     real(c_double) :: val
     integer(c_int64_t) :: idx
  end type pair

contains

  pure function better(a, b) result(r)
    type(pair), intent(in) :: a, b
    type(pair) :: r
    if (a%val > b%val) then
       r = a
    else if (a%val < b%val) then
       r = b
    else
       if (a%idx < b%idx) then
          r = a
       else
          r = b
       end if
    end if
  end function better
end module pair_mod

subroutine tsvc_2_s3110_fp64(aa, bb, LEN_2D) bind(c, name='tsvc_2_s3110_fp64')
  use iso_c_binding, only: c_double, c_int64_t
  use pair_mod, only: pair, better
  use omp_lib, only: omp_get_max_threads, omp_get_thread_num
  implicit none
  integer(c_int64_t), value :: LEN_2D
  real(c_double), intent(in) :: aa(LEN_2D*LEN_2D)
  real(c_double), intent(out) :: bb(*)

  integer(c_int64_t), parameter :: VL = 8_c_int64_t
  integer(c_int64_t), parameter :: SERIAL_LIMIT = 65536_c_int64_t

  integer(c_int64_t) :: k, kk, n, xindex, yindex, remain
  integer :: t, nt, tid
  type(pair), allocatable :: parts(:)
  type(pair) :: local, best

  real(c_double) :: pmax(VL), v
  integer(c_int64_t) :: pidx(VL)

  n = LEN_2D * LEN_2D

  if (n <= SERIAL_LIMIT) then
     best%val = aa(1)
     best%idx = 1_c_int64_t
     do k = 2_c_int64_t, n
        if (aa(k) > best%val) then
           best%val = aa(k)
           best%idx = k
        end if
     end do
  else
     nt = omp_get_max_threads()
     allocate(parts(nt))
     local%val = -huge(1.0_c_double)
     local%idx = huge(1_c_int64_t)
     parts(:) = local

     !$omp parallel default(none) shared(aa, n, parts) private(k, kk, tid, pmax, pidx, v, local)
     tid = omp_get_thread_num() + 1
     pmax = -huge(1.0_c_double)
     pidx = huge(1_c_int64_t)
     !$omp do schedule(static)
     do k = 1_c_int64_t, n - VL + 1_c_int64_t, VL
        !$omp simd
        do kk = 0_c_int64_t, VL - 1_c_int64_t
           v = aa(k + kk)
           if (v > pmax(kk + 1_c_int64_t)) then
              pmax(kk + 1_c_int64_t) = v
              pidx(kk + 1_c_int64_t) = k + kk
           end if
        end do
        !$omp end simd
     end do
     !$omp end do

     local%val = maxval(pmax)
     local%idx = minval(pidx, mask=(pmax == local%val))
     parts(tid) = local
     !$omp end parallel

     remain = mod(n, VL)
     if (remain /= 0_c_int64_t) then
        k = n - remain + 1_c_int64_t
        best%val = aa(k)
        best%idx = k
        do k = n - remain + 2_c_int64_t, n
           if (aa(k) > best%val) then
              best%val = aa(k)
              best%idx = k
           end if
        end do
        best = better(best, parts(1))
     else
        best = parts(1)
     end if

     do t = 2, nt
        if (parts(t)%val > best%val .or. &
            (parts(t)%val == best%val .and. parts(t)%idx < best%idx)) then
           best%val = parts(t)%val
           best%idx = parts(t)%idx
        end if
     end do
  end if

  xindex = (best%idx - 1_c_int64_t) / LEN_2D
  yindex = mod(best%idx - 1_c_int64_t, LEN_2D)
  bb(1) = best%val + real(xindex, c_double) + real(yindex, c_double)
end subroutine tsvc_2_s3110_fp64
