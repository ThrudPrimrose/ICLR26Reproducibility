subroutine argmax_with_index_fp64(a, out_index, out_value, len_1d) bind(c, name='argmax_with_index_fp64')
  use, intrinsic :: iso_c_binding
  use, intrinsic :: omp_lib
  implicit none
  type(c_ptr), value :: a, out_index, out_value
  integer(c_int64_t), value :: len_1d

  real(c_double), contiguous, dimension(:), pointer :: x
  real(c_double), pointer, dimension(:) :: ov
  integer(c_int64_t), pointer, dimension(:) :: oi
  real(c_double), save, dimension(1024) :: tmax
  integer(c_int64_t), save, dimension(1024) :: tidx
  real(c_double) :: best, mx
  integer(c_int64_t) :: best_i, n, lo0, hi0, i, mi, t, nthreads

  call c_f_pointer(a, x, [int(len_1d, c_size_t)])
  call c_f_pointer(out_index, oi, [1])
  call c_f_pointer(out_value, ov, [1])
  n = len_1d
  if (n <= 0) return
  if (n == 1) then
     ov(1) = x(1)
     oi(1) = 1
     return
  end if
  if (x(1) /= x(1)) then   ! a[0] NaN: reference keeps it
     ov(1) = x(1)
     oi(1) = 1
     return
  end if

  do t = 1, 1024
     tmax(t) = -inf_real()
     tidx(t) = 0
  end do

  !$omp parallel private(t, lo0, hi0, i, mx, mi) shared(x, n, tmax, tidx, nthreads)
  !$omp single
  nthreads = omp_get_num_threads()
  !$omp end single
  !$omp barrier
  t = omp_get_thread_num() + 1
  if (t <= 1024) then
     lo0 = int(n * (t - 1), c_int64_t) / int(nthreads, c_int64_t)
     hi0 = int(n * t, c_int64_t) / int(nthreads, c_int64_t)
     if (hi0 > lo0) then
        mx = x(lo0 + 1); mi = lo0
        do i = lo0 + 1, hi0 - 1
           if (x(i + 1) > mx) then
              mx = x(i + 1); mi = i
           end if
        end do
        tmax(t) = mx
        tidx(t) = mi
     end if
  end if
  !$omp end parallel

  best = -inf_real(); best_i = 0
  do t = 1, nthreads
     if (tmax(t) > best .or. (tmax(t) == best .and. tidx(t) < best_i)) then
        best = tmax(t); best_i = tidx(t)
     end if
  end do

  if (best == 0.0d0) then   ! +/-0.0 tie-breaking: re-run exact reference semantics
     mx = x(1); mi = 0
     do i = 2, n
        if (x(i) > mx) then
           mx = x(i); mi = i - 1
        end if
     end do
     best = mx; best_i = mi
  end if

  ov(1) = best
  oi(1) = best_i + 1   ! Fortran 1-based position (C reference is 0-based)
contains
  pure function inf_real() result(r)
     real(c_double) :: r
     r = huge(1.0d0)
  end function
end subroutine
