subroutine ext_war_unit_fp64(a, b, len_1d, workspace, workspace_size) bind(C)
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: len_1d, workspace_size
  real(c_double), intent(inout) :: a(len_1d)
  real(c_double), intent(in) :: b(len_1d)
  type(c_ptr), value :: workspace
  integer(c_int64_t), parameter :: BS = 2048
  integer(c_int64_t) :: i, t, nt, lo, hi, l, r, nbuf
  real(c_double) :: boundary
  real(c_double) :: buf(BS + 1)

  if (len_1d <= 1) return

  if (len_1d <= 1024) then
    do i = 1, len_1d - 1
      a(i) = a(i + 1) + b(i)
    end do
    return
  end if

  !$omp parallel private(t, nt, lo, hi, l, r, nbuf, i, boundary, buf) shared(a, b, len_1d)
    nt = omp_get_num_threads()
    t = omp_get_thread_num()
    lo = (len_1d * t) / nt + 1
    hi = (len_1d * (t + 1)) / nt

    if (hi < len_1d) then
      boundary = a(hi + 1)
    else
      boundary = 0.0_c_double
    end if

    !$omp barrier

    if (lo <= hi) then
      do l = lo, hi, BS
        r = min(l + BS - 1, hi)
        nbuf = r - l + 1

        do i = l, r
          buf(i - l + 1) = a(i)
        end do

        if (r < len_1d) then
          if (r < hi) then
            buf(nbuf + 1) = a(r + 1)
          else
            buf(nbuf + 1) = boundary
          end if
        end if

        do i = l, min(r, len_1d - 1)
          a(i) = buf(i - l + 2) + b(i)
        end do
      end do
    end if
  !$omp end parallel
end subroutine ext_war_unit_fp64
