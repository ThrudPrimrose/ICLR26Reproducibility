module wf_diff_skew_mod
  use, intrinsic :: iso_c_binding, only: c_double, c_int64_t
  use omp_lib
  implicit none
  integer, parameter :: PAD = 8
contains
  subroutine wf_diff_skew_fp64(a, LEN_2D) bind(c, name="wf_diff_skew_fp64")
    real(c_double), intent(inout) :: a(*)
    integer(c_int64_t), value, intent(in) :: LEN_2D
    integer(c_int64_t) :: n, m, i, j, bp, bc, j0, j1, chunk, rem, val
    integer :: nt_max, nt, tid
    integer(c_int64_t), allocatable :: ready(:,:)

    n = LEN_2D
    if (n <= 1_c_int64_t) return

    nt_max = omp_get_max_threads()
    allocate(ready(0:PAD-1, 0:nt_max-1))
    ready = 0_c_int64_t

    !$omp parallel default(none) shared(a, n, ready) private(nt, tid, i, j, bp, bc, j0, j1, chunk, rem, val, m)
    nt = omp_get_num_threads()
    tid = omp_get_thread_num()
    m = n - 1_c_int64_t
    chunk = m / int(nt, c_int64_t)
    rem = mod(m, int(nt, c_int64_t))
    if (tid < nt - 1) then
      if (tid < rem) then
        j0 = int(tid, c_int64_t) * (chunk + 1_c_int64_t)
        j1 = j0 + chunk
      else
        j0 = int(rem, c_int64_t) * (chunk + 1_c_int64_t) + int(tid - rem, c_int64_t) * chunk
        j1 = j0 + chunk - 1_c_int64_t
      end if
    else
      ! rightmost thread also owns up to n-2
      if (tid < rem) then
        j0 = int(tid, c_int64_t) * (chunk + 1_c_int64_t)
        j1 = j0 + chunk
      else
        j0 = int(rem, c_int64_t) * (chunk + 1_c_int64_t) + int(tid - rem, c_int64_t) * chunk
        j1 = j0 + chunk - 1_c_int64_t
      end if
    end if

    do i = 1_c_int64_t, n - 1_c_int64_t
      if (tid < nt - 1) then
        do
          !$omp atomic read
          val = ready(0, tid + 1)
          !$omp end atomic
          if (val >= i - 1_c_int64_t) exit
        end do
      end if

      if (j0 <= j1) then
        bp = (i - 1_c_int64_t) * n
        bc = i * n
        !$omp simd
        do j = j0, j1
          a(bc + j + 1_c_int64_t) = a(bc + j + 1_c_int64_t) &
                                  + a(bp + j + 1_c_int64_t) &
                                  + a(bp + j + 2_c_int64_t)
        end do
        !$omp end simd
      end if

      !$omp atomic write
      ready(0, tid) = i
      !$omp end atomic
      !$omp flush
    end do
    !$omp end parallel

    deallocate(ready)
  end subroutine wf_diff_skew_fp64
end module wf_diff_skew_mod
