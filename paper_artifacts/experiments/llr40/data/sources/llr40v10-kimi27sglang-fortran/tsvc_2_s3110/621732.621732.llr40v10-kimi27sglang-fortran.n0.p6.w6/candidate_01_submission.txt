module tsvc_2_s3110_mod
  use iso_c_binding, only: c_double, c_int64_t
  use omp_lib, only: omp_get_max_threads
  implicit none
contains
  subroutine tsvc_2_s3110_fp64(aa, bb, LEN_2D) bind(c, name="tsvc_2_s3110_fp64")
    implicit none
    real(c_double), intent(in) :: aa(0:*)
    real(c_double), intent(out) :: bb(0:*)
    integer(c_int64_t), value, intent(in) :: LEN_2D
    integer(c_int64_t) :: n, n2, nt, t, pos, i, j, k, chunk, start, finish
    real(c_double) :: maxv, v
    real(c_double), allocatable :: tmax(:)
    integer(c_int64_t), allocatable :: tpos(:)

    n = LEN_2D
    n2 = n * n
    nt = omp_get_max_threads()
    allocate(tmax(0:nt-1), tpos(0:nt-1))

    !$omp parallel do schedule(static, 1) private(k, v, start, finish, chunk)
    do t = 0, nt - 1
      chunk = (n2 + nt - 1) / nt
      start = t * chunk
      finish = min(start + chunk, n2) - 1
      if (start <= finish) then
        tmax(t) = aa(start)
        tpos(t) = start
        do k = start, finish
          v = aa(k)
          if (v > tmax(t)) then
            tmax(t) = v
            tpos(t) = k
          end if
        end do
      else
        tmax(t) = -huge(1.0_c_double)
        tpos(t) = 0
      end if
    end do

    maxv = tmax(0)
    pos = tpos(0)
    do t = 1, nt - 1
      if (tmax(t) > maxv) then
        maxv = tmax(t)
        pos = tpos(t)
      end if
    end do

    i = pos / n
    j = pos - i * n
    bb(0) = maxv + real(i, c_double) + real(j, c_double)
  end subroutine tsvc_2_s3110_fp64
end module tsvc_2_s3110_mod
