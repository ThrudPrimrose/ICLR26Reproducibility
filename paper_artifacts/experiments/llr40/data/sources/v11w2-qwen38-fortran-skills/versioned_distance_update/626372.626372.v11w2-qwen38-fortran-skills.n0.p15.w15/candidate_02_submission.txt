subroutine versioned_distance_update(a, b, c, k, len_1d, workspace, workspace_size) bind(C, name='versioned_distance_update_fp64')
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: k, len_1d, workspace_size
  real(c_double), intent(inout) :: a(len_1d)
  real(c_double), intent(in) :: b(len_1d), c(len_1d)
  integer(c_int8_t), intent(inout) :: workspace(workspace_size)
  integer(c_int64_t) :: i, r0, r1, s, j, jhi, smax, ng, n, hi
  integer(kind=8) :: g

  n = len_1d
  if (k < 0) return
  if (k == 0) then
    do i = 1, n
      a(i) = 0.75d0 * a(i) + b(i) * c(i)
    end do
    return
  end if
  if (n <= k) return

  if (k == 1) then
    do i = 2, n
      a(i) = 0.75d0 * a(i - 1) + b(i) * c(i)
    end do
    return
  end if
  if (k == 2) then
    do i = 3, n
      a(i) = 0.75d0 * a(i - 2) + b(i) * c(i)
    end do
    return
  end if
  if (k == 3) then
    do i = 4, n
      a(i) = 0.75d0 * a(i - 3) + b(i) * c(i)
    end do
    return
  end if
  if (k == 4) then
    do i = 5, n
      a(i) = 0.75d0 * a(i - 4) + b(i) * c(i)
    end do
    return
  end if
  if (k == 5) then
    do i = 6, n
      a(i) = 0.75d0 * a(i - 5) + b(i) * c(i)
    end do
    return
  end if
  if (k == 6) then
    do i = 7, n
      a(i) = 0.75d0 * a(i - 6) + b(i) * c(i)
    end do
    return
  end if
  if (k == 7) then
    do i = 8, n
      a(i) = 0.75d0 * a(i - 7) + b(i) * c(i)
    end do
    return
  end if
  if (k == 8) then
    do i = 9, n
      a(i) = 0.75d0 * a(i - 8) + b(i) * c(i)
    end do
    return
  end if

  ! General K >= 9.
  ! First element of every chain (i = k+1 .. min(2k, n)) reads only seed
  ! values -> fully parallel.
  hi = min(2 * k, n)
  if (n - k > 1000000) then
    !$omp parallel do schedule(static)
    do i = k + 1, hi
      a(i) = 0.75d0 * a(i - k) + b(i) * c(i)
    end do
    ! Chain region, transposed: group g holds 8 consecutive residues; each
    ! diagonal step writes 8 consecutive elements (independent within the
    ! step), which vectorizes, while the 8 chains advance in lockstep.
    ng = (k + 7) / 8
    !$omp parallel do schedule(static)
    do g = 1, ng
      r0 = (g - 1) * 8 + 1
      r1 = min(g * 8, k)
      smax = (n - r0) / k
      do s = 2, smax
        jhi = r1 + s * k
        if (jhi > n) jhi = n
        !$omp simd
        do j = r0 + s * k, jhi
          a(j) = 0.75d0 * a(j - k) + b(j) * c(j)
        end do
      end do
    end do
  else
    do i = k + 1, hi
      a(i) = 0.75d0 * a(i - k) + b(i) * c(i)
    end do
    ng = (k + 7) / 8
    do g = 1, ng
      r0 = (g - 1) * 8 + 1
      r1 = min(g * 8, k)
      smax = (n - r0) / k
      do s = 2, smax
        jhi = r1 + s * k
        if (jhi > n) jhi = n
        !$omp simd
        do j = r0 + s * k, jhi
          a(j) = 0.75d0 * a(j - k) + b(j) * c(j)
        end do
      end do
    end do
  end if
end subroutine versioned_distance_update
