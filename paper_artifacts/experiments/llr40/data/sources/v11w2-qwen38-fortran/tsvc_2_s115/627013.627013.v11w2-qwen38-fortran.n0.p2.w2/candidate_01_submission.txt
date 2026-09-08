subroutine tsvc_2_s115_fp64(a, aa, LEN_2D) bind(C, name="tsvc_2_s115_fp64")
  use, intrinsic :: iso_c_binding
  use omp_lib
  implicit none
  type(c_ptr), intent(in), value :: a, aa
  integer(c_int64_t), value :: LEN_2D
  real(c_double), dimension(:),  pointer :: av
  real(c_double), dimension(:,:), pointer :: B
  integer :: n, np, nblk, blk, r, lo, hi, i, j, p, pidx
  real(c_double) :: x, s
  real(c_double), dimension(:,:), allocatable :: pp

  n = int(LEN_2D, kind=4)
  if (n <= 1) return
  call c_f_pointer(a, av, [n])
  call c_f_pointer(aa, B, [n, n])

  if (n < 2048) then
    do j = 1, n
      x = av(j)
      do i = j+1, n
        av(i) = av(i) - B(i, j)*x
      end do
    end do
    return
  end if

  np = omp_get_max_threads()
  blk = 512
  nblk = (n + blk - 1) / blk
  allocate(pp(blk+1, np))

  !$omp parallel private(pidx, r, lo, hi, i, j, p, x, s)
  pidx = omp_get_thread_num() + 1
  do r = 0, nblk-1
    lo = r*blk + 1
    hi = min((r+1)*blk, n)
    if (lo > 1) then
      pp(1:hi-lo+1, pidx) = 0.0d0
      !$omp do
      do j = 1, lo-1
        x = av(j)
        do i = lo, hi
          pp(i-lo+1, pidx) = pp(i-lo+1, pidx) + B(i, j)*x
        end do
      end do
      !$omp end do
      !$omp do
      do i = lo, hi
        s = 0.0d0
        do p = 1, np
          s = s + pp(i-lo+1, p)
        end do
        av(i) = av(i) - s
      end do
      !$omp end do
    end if
    !$omp master
    do j = lo, hi-1
      x = av(j)
      do i = j+1, hi
        av(i) = av(i) - B(i, j)*x
      end do
    end do
    !$omp end master
    !$omp barrier
  end do
  !$omp end parallel
end subroutine tsvc_2_s115_fp64
