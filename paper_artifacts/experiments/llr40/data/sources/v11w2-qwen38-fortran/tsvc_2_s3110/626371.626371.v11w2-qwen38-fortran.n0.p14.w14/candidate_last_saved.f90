subroutine tsvc_2_s3110_fp64(aa, bb, len_2d) bind(C, name="tsvc_2_s3110_fp64")
  use iso_c_binding
  use omp_lib
  use, intrinsic :: ieee_arithmetic
  implicit none
  type(c_ptr), intent(in), value :: aa
  type(c_ptr), intent(in), value :: bb
  integer(c_int64_t), intent(in), value :: len_2d

  integer(c_int64_t) :: n, total, i, k, s, e, s2, e2, lp
  real(c_double), pointer :: a(:)
  real(c_double), pointer :: b(:)
  real(c_double) :: maxv, chksum, lm, ninf
  integer(c_int64_t) :: best_pos, T
  real(c_double), allocatable :: tmax(:)
  integer(c_int64_t), allocatable :: tpos(:)
  integer :: nt, tid

  ninf = ieee_value(1.0d0, IEEE_NEGATIVE_INF)
  n = len_2d
  total = n * n
  call c_f_pointer(aa, a, [total])
  call c_f_pointer(bb, b, [4])

  nt = omp_get_max_threads()
  allocate(tmax(nt), tpos(nt))
  tmax(:) = ninf
  tpos(:) = huge(1_8)

  ! Pass 1: per-thread vectorized maxval
  !$omp parallel default(none) &
  !$omp& shared(a, total, nt, tmax, tpos) &
  !$omp& private(i, s, e, lm, tid)
  tid = omp_get_thread_num()
  s = 1 + total * int(tid, 8) / nt
  e = 1 + total * int(tid + 1, 8) / nt - 1
  if (s <= e) then
    lm = maxval(a(s:e))
    tmax(tid + 1) = lm
    tpos(tid + 1) = s
  end if
  !$omp end parallel

  ! Combine: global max value + earliest chunk T whose max == maxv
  maxv = ninf
  do i = 1, nt
    if (tmax(i) > maxv) then
      maxv = tmax(i)
    end if
  end do
  T = 1
  do i = 1, nt
    if (tmax(i) == maxv) then
      T = i
      exit
    end if
  end do

  ! Pass 2: parallel search for first occurrence of maxv in chunk T
  s = 1 + total * int(T - 1, 8) / nt
  e = 1 + total * int(T, 8) / nt - 1
  tpos(:) = huge(1_8)
  !$omp parallel default(none) &
  !$omp& shared(a, s, e, nt, tpos, maxv) &
  !$omp& private(i, s2, e2, lp, tid)
  tid = omp_get_thread_num()
  s2 = s + (e - s + 1) * int(tid, 8) / nt
  e2 = s + (e - s + 1) * int(tid + 1, 8) / nt - 1
  if (s2 <= e2) then
    lp = e2 + 1
    do i = s2, e2
      if (a(i) == maxv) then
        lp = i
        exit
      end if
    end do
    tpos(tid + 1) = lp
  end if
  !$omp end parallel

  best_pos = huge(1_8)
  do i = 1, nt
    if (tpos(i) < best_pos) best_pos = tpos(i)
  end do

  k = best_pos - 1
  chksum = maxv + dble(k / n) + dble(mod(k, n))
  b(1) = chksum

end subroutine tsvc_2_s3110_fp64
