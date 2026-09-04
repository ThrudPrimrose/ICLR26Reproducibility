module ebc_state
  use iso_c_binding
  implicit none
  integer(c_int64_t), target, dimension(1) :: best_target = [0_8]
end module ebc_state

subroutine ext_break_capture_fp64(arr, out_index, out_value, len_1d) bind(C, name="ext_break_capture_fp64")
  use iso_c_binding
  use ebc_state
  implicit none
  integer(c_int64_t),  value, intent(in)  :: len_1d
  double precision, dimension(len_1d), intent(in) :: arr
  integer(c_int64_t), intent(out) :: out_index
  real(c_double),  intent(out) :: out_value

  interface
    function omp_get_num_threads_c() bind(C, name="omp_get_num_threads")
      use iso_c_binding
      integer(c_int) omp_get_num_threads_c
    end function
    function omp_get_thread_num_c() bind(C, name="omp_get_thread_num")
      use iso_c_binding
      integer(c_int) omp_get_thread_num_c
    end function
  end interface

  integer(c_int64_t) :: n, Csz, nchunks, g, T, lo, hi, i, found, b, best
  integer(c_int64_t), pointer :: bestp(:)
  real(c_double) :: one

  n = len_1d
  one = 1.0d0

  if (n < ishft(1_8, 13)) then
    out_index = -1
    out_value = -1.0d0
    do i = 1, n
      if (arr(i) > one) then
        out_index = i
        out_value = arr(i)
        return
      end if
    end do
    return
  end if

  bestp => best_target
  bestp(1) = huge(0_8)

  !$omp parallel default(none) shared(arr, n, bestp, one) private(T, Csz, nchunks, g, lo, hi, i, found, b)
  T = int(omp_get_num_threads_c(), c_int64_t)
  Csz = (n + 16*T - 1) / (16*T)
  Csz = max(Csz, 64_8)
  nchunks = (n + Csz - 1) / Csz

  g = int(omp_get_thread_num_c(), c_int64_t)
  do
    if (g >= nchunks) exit
    b = bestp(1)
    if (b < huge(0_8) .and. g >= b / Csz) exit
    lo = g*Csz + 1
    hi = min(n, (g+1)*Csz)
    found = 0_8
    do i = lo, hi
      if (arr(i) > one) then
        found = i
        exit
      end if
    end do
    if (found > 0_8) then
      !$omp critical
      if (found < bestp(1)) bestp(1) = found
      !$omp end critical
    end if
    g = g + T
  end do
  !$omp end parallel

  best = bestp(1)
  if (best < huge(0_8)) then
    out_index = best
    out_value = arr(best)
  else
    out_index = -1
    out_value = -1.0d0
  end if
end subroutine ext_break_capture_fp64
