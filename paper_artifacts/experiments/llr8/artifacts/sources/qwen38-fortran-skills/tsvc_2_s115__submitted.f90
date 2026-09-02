subroutine tsvc_2_s115_fp64(a, aa, len_2d, workspace, workspace_size) bind(C, name="tsvc_2_s115_fp64")
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: len_2d, workspace_size
  real(c_double), intent(inout) :: a(len_2d)
  real(c_double), intent(in)    :: aa(len_2d, len_2d)
  real(c_double), intent(inout) :: workspace(workspace_size)
  integer :: n, b, k, bs, he, ks, ke, i, j, nblk, sblk, nt, t
  real(c_double) :: s
  real(c_double), dimension(:), allocatable :: ybuf

  n = int(len_2d)
  if (n <= 1) return

  if (n <= 4096) then
    ! serial path: unit-stride inner loop, vectorizes
    do j = 1, n - 1
      do i = j + 1, n
        a(i) = a(i) - aa(i, j) * a(j)
      end do
    end do
    return
  end if

  nblk = 2048
  sblk = 128

  nt = omp_get_max_threads()
  allocate(ybuf(nt * nblk))
  ybuf = 0.0d0

  !$omp parallel
  do b = 1, (n + nblk - 1) / nblk
    bs = (b - 1) * nblk + 1
    he = min(b * nblk, n)
    if (bs > 1) then
      ! inter-block: distribute j over threads; each thread accumulates
      ! into its private y_t(i) for i in the block (unit stride in i)
      !$omp do
      do j = 1, bs - 1
        t = omp_get_thread_num()
        do i = bs, he
          ybuf(t * nblk + i - bs + 1) = ybuf(t * nblk + i - bs + 1) + aa(i, j) * a(j)
        end do
      end do
      ! combine and zero
      !$omp do
      do i = bs, he
        s = a(i)
        do t = 0, nt - 1
          s = s - ybuf(t * nblk + i - bs + 1)
          ybuf(t * nblk + i - bs + 1) = 0.0d0
        end do
        a(i) = s
      end do
    end if
    ! forward substitution inside the block, in sub-blocks:
    ! each sub-block after the first gets a parallel y-accum over the
    ! earlier sub-blocks' j's, then a small serial sweep.
    do k = 1, (he - bs + 1 + sblk - 1) / sblk
      ks = bs + (k - 1) * sblk
      ke = min(bs + k * sblk - 1, he)
      if (ks > bs) then
        !$omp do
        do j = bs, ks - 1
          t = omp_get_thread_num()
          do i = ks, ke
            ybuf(t * sblk + i - ks + 1) = ybuf(t * sblk + i - ks + 1) + aa(i, j) * a(j)
          end do
        end do
        !$omp do
        do i = ks, ke
          s = a(i)
          do t = 0, nt - 1
            s = s - ybuf(t * sblk + i - ks + 1)
            ybuf(t * sblk + i - ks + 1) = 0.0d0
          end do
          a(i) = s
        end do
      end if
      !$omp single
      do j = ks, ke - 1
        do i = j + 1, ke
          a(i) = a(i) - aa(i, j) * a(j)
        end do
      end do
      !$omp end single
    end do
  end do
  !$omp end parallel
  deallocate(ybuf)
end subroutine tsvc_2_s115_fp64
